#!/usr/bin/env python3
"""Synchronize GitHub issues from the generated support matrix.

The tool owns only issues that contain a stable
``crossgl-support-issue-sync`` marker. It creates one parent issue per
backend plus one frontend parent, creates sub-issues for every current support
backlog row, updates existing managed issues when matrix data changes, and
closes managed backlog issues that disappear from the matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib import error
from urllib import parse
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = ROOT / "support" / "generated" / "support-matrix.json"
DEFAULT_SIGNALS_PATH = ROOT / "support" / "generated" / "support-signals.json"
DEFAULT_MATRIX_CHECK_REPORT_PATH = (
    ROOT / "support" / "generated" / "support-matrix-check.json"
)
SUPPORT_MATRIX_GENERATOR = "tools/support_matrix.py"
SUPPORT_MATRIX_SCHEMA_VERSION = 1
SUPPORT_MATRIX_REQUIRED_FIELDS = (
    "schema_version",
    "generator",
    "summary",
    "backends",
    "features",
    "backlog",
)
SUPPORT_SIGNALS_GENERATOR = "tools/support_signals.py"
SUPPORT_SIGNALS_SCHEMA_VERSION = 1
SUPPORT_SIGNALS_REQUIRED_FIELDS = (
    "schema_version",
    "generator",
    "summary",
    "features",
    "issues",
)
MATRIX_CHECK_REPORT_GENERATOR = "tools/support_matrix.py check"
MATRIX_CHECK_REPORT_SCHEMA_VERSION = 1
MATRIX_CHECK_REPORT_REQUIRED_FIELDS = (
    "schema_version",
    "generator",
    "ok",
    "summary",
    "artifacts",
)

API_VERSION = "2026-03-10"
MARKER_NAME = "crossgl-support-issue-sync"
MARKER_RE = re.compile(r"<!--\s*{}:\s*([^>\s]+)\s*-->".format(MARKER_NAME))

LABEL_MANAGED = "support:matrix"
LABEL_PARENT = "support:parent"
LABEL_BACKLOG = "support:backlog"
LABEL_EXTRACTED = "support:extracted"
LABEL_PREFIX_BACKEND = "support-backend:"
LABEL_PREFIX_CATEGORY = "support-category:"
LABEL_PREFIX_STATUS = "support-status:"
PLANNED_ACTION_SAMPLE_LIMIT = 12

FRONTEND_ID = "frontend"
FRONTEND_NAME = "Frontend / IR / Parser"

BACKLOG_STATUSES = {
    "partial",
    "diagnostic",
    "validated_rejection",
    "unsupported",
    "unknown",
}


class GitHubApiError(RuntimeError):
    """Raised when GitHub returns an unexpected API error."""

    def __init__(
        self,
        method: str,
        path: str,
        status: int,
        body: str,
        headers: dict[str, str] | None = None,
    ):
        super().__init__("{} {} failed with {}: {}".format(method, path, status, body))
        self.method = method
        self.path = path
        self.status = status
        self.body = body
        self.headers = headers or {}


class SupportIssueSyncMutationError(RuntimeError):
    """Raised when GitHub mutation fails after sync has started."""

    def __init__(
        self,
        phase: str,
        operation: dict[str, Any],
        summary: dict[str, int],
        cause: Exception,
        operation_ledger: list[dict[str, Any]] | None = None,
    ):
        super().__init__("support issue sync failed during {}: {}".format(phase, cause))
        self.phase = phase
        self.operation = operation
        self.summary = dict(summary)
        self.cause = cause
        self.operation_ledger = list(operation_ledger or [])


class SupportIssueSyncPreflightError(RuntimeError):
    """Raised when GitHub read/preflight inspection fails before mutation."""

    def __init__(
        self,
        phase: str,
        operation: dict[str, Any],
        cause: Exception,
    ):
        super().__init__(
            "support issue sync preflight failed during {}: {}".format(phase, cause)
        )
        self.phase = phase
        self.operation = operation
        self.cause = cause


@dataclass(frozen=True)
class DesiredIssue:
    key: str
    title: str
    body: str
    labels: tuple[str, ...]
    parent_key: str | None = None


def load_matrix(path: Path) -> dict[str, Any]:
    return load_json_input(
        path,
        required=True,
        expected_generator=SUPPORT_MATRIX_GENERATOR,
        required_fields=SUPPORT_MATRIX_REQUIRED_FIELDS,
        schema_version=SUPPORT_MATRIX_SCHEMA_VERSION,
    )


def load_signals(path: Path | None) -> dict[str, Any] | None:
    return load_json_input(
        path,
        required=False,
        expected_generator=SUPPORT_SIGNALS_GENERATOR,
        required_fields=SUPPORT_SIGNALS_REQUIRED_FIELDS,
        schema_version=SUPPORT_SIGNALS_SCHEMA_VERSION,
    )


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return optional_json_load_error(path, type(exc).__name__, str(exc))
    if not isinstance(data, dict):
        return optional_json_load_error(
            path,
            "InvalidReportType",
            "expected JSON object, got {}".format(type(data).__name__),
        )
    return data


def optional_json_load_error(
    path: Path | None,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "load_error": {
            "path": str(path) if path is not None else None,
            "type": error_type,
            "message": message,
        }
    }


def load_json_input(
    path: Path | None,
    *,
    required: bool,
    expected_generator: str | None = None,
    required_fields: tuple[str, ...] = (),
    schema_version: int | None = None,
) -> dict[str, Any] | None:
    if path is None or not path.exists():
        if not required:
            return None
        return optional_json_load_error(
            path, "MissingInput", "required input not found"
        )

    data = load_optional_json(path)
    if data is None or data.get("load_error"):
        return data

    schema_error = json_report_schema_error(
        data,
        path,
        expected_generator=expected_generator,
        required_fields=required_fields,
        schema_version=schema_version,
    )
    if schema_error is not None:
        return {"load_error": schema_error}
    return data


def json_report_schema_error(
    report: dict[str, Any],
    path: Path | None = None,
    *,
    expected_generator: str | None = None,
    required_fields: tuple[str, ...] = (),
    schema_version: int | None = None,
) -> dict[str, Any] | None:
    missing_fields = [field for field in required_fields if field not in report]
    if missing_fields:
        return optional_json_load_error(
            path,
            "MissingReportFields",
            "missing required fields: {}".format(", ".join(missing_fields)),
        )["load_error"]

    if schema_version is not None:
        actual_schema_version = report.get("schema_version")
        if actual_schema_version != schema_version:
            return optional_json_load_error(
                path,
                "UnsupportedSchemaVersion",
                "expected schema_version {}, got {}".format(
                    schema_version,
                    actual_schema_version,
                ),
            )["load_error"]

    if expected_generator is not None:
        actual_generator = report.get("generator")
        if actual_generator != expected_generator:
            return optional_json_load_error(
                path,
                "UnexpectedReportGenerator",
                "expected generator {}, got {}".format(
                    expected_generator,
                    actual_generator,
                ),
            )["load_error"]

    return None


def input_load_error(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    return report.get("load_error")


def input_failure_summary(
    input_name: str,
    path: Path | None,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input": input_name,
        "path": str(path) if path is not None else None,
        "error": dict(report.get("load_error", {})),
    }


def marker_for(key: str) -> str:
    return "<!-- {}: {} -->".format(MARKER_NAME, key)


def marker_key(body: str | None) -> str | None:
    if not body:
        return None
    match = MARKER_RE.search(body)
    if not match:
        return None
    return match.group(1)


def compact_label_part(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    return value.strip("-") or "unknown"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(cell).replace("\n", " ").strip() for cell in row)
            + " |"
        )
    return "\n".join(lines)


def feature_lookup(matrix: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {}
    for feature in matrix.get("features", []):
        feature_id = feature["id"]
        for backend_id, entry in feature.get("support", {}).items():
            lookup[(backend_id, feature_id)] = {
                "feature": feature,
                "support": entry,
            }
    return lookup


def signal_lookup(
    signals: dict[str, Any] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {}
    if not signals:
        return lookup
    for feature in signals.get("features", []):
        feature_id = feature["id"]
        for backend_id, entry in feature.get("support", {}).items():
            lookup[(backend_id, feature_id)] = entry
    return lookup


def backlog_by_backend(matrix: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        backend["id"]: [] for backend in matrix.get("backends", [])
    }
    for item in matrix.get("backlog", []):
        rows.setdefault(item["backend_id"], []).append(item)
    for backend_rows in rows.values():
        backend_rows.sort(key=lambda item: (item["category"], item["feature_id"]))
    return rows


def parent_body(
    matrix: dict[str, Any], backend: dict[str, Any], rows: list[dict[str, Any]]
) -> str:
    backend_id = backend["id"]
    counts = matrix["summary"]["status_counts"].get(backend_id, {})
    summary_rows = [
        [status, counts.get(status, 0)]
        for status in (
            "supported",
            "partial",
            "diagnostic",
            "validated_rejection",
            "unsupported",
            "unknown",
        )
    ]
    backlog_rows = [
        [item["category"], item["feature"], item["status"]] for item in rows[:80]
    ]
    if not backlog_rows:
        backlog_text = (
            "No current backend backlog rows are present in the support matrix."
        )
    else:
        backlog_text = markdown_table(["Category", "Feature", "Status"], backlog_rows)
        if len(rows) > len(backlog_rows):
            backlog_text += (
                "\n\nAdditional backlog rows omitted from this summary: {}".format(
                    len(rows) - len(backlog_rows)
                )
            )

    docs = backend.get("docs") or []
    docs_text = "\n".join("- [{}]({})".format(doc["name"], doc["url"]) for doc in docs)
    if not docs_text:
        docs_text = "- No documentation URLs are configured."

    return "\n\n".join(
        [
            marker_for("parent:{}".format(backend_id)),
            "# {} Support Coverage".format(backend["name"]),
            "This issue is managed from `support/generated/support-matrix.json` by `tools/sync_support_issues.py`.",
            "Open backlog rows: **{}**".format(len(rows)),
            markdown_table(["Status", "Count"], summary_rows),
            "## Current Backlog\n\n{}".format(backlog_text),
            "## Documentation Sources\n\n{}".format(docs_text),
        ]
    )


def frontend_parent_body() -> str:
    return "\n\n".join(
        [
            marker_for("parent:{}".format(FRONTEND_ID)),
            "# {} Support Coverage".format(FRONTEND_NAME),
            "This issue is managed by `tools/sync_support_issues.py`.",
            "The current support matrix is backend-oriented, so there are no frontend-specific backlog sub-issues to create yet.",
            "Frontend, parser, IR, validation, and shared source-language work should be added to the support catalog before it can produce generated child issues here.",
        ]
    )


def child_body(
    item: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
    signals_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> str:
    entry = lookup.get((item["backend_id"], item["feature_id"]), {})
    support = entry.get("support", {})
    feature = entry.get("feature", {})
    evidence = support.get("evidence") or []
    evidence_text = "\n".join("- `{}`".format(value) for value in evidence)
    if not evidence_text:
        evidence_text = "- No evidence is recorded for this non-supported row."
    notes = item.get("notes") or support.get("notes") or "No notes recorded."
    signal_text = format_signal_section(
        signals_lookup.get((item["backend_id"], item["feature_id"]))
        if signals_lookup
        else None
    )

    return "\n\n".join(
        [
            marker_for("backlog:{}:{}".format(item["backend_id"], item["feature_id"])),
            "# {}: {}".format(item["backend"], item["feature"]),
            "This sub-issue is managed from `support/generated/support-matrix.json` by `tools/sync_support_issues.py`.",
            markdown_table(
                ["Field", "Value"],
                [
                    ["Backend", item["backend"]],
                    ["Feature ID", item["feature_id"]],
                    ["Category", item["category"]],
                    ["Status", item["status"]],
                ],
            ),
            "## Feature Description\n\n{}".format(
                feature.get("description", "No feature description recorded.")
            ),
            "## Current Gap\n\n{}".format(notes),
            "## Recorded Evidence\n\n{}".format(evidence_text),
            "## Extracted Signals\n\n{}".format(signal_text),
            "## Completion Rule\n\nWhen this matrix row becomes `supported`, the next sync will close this managed sub-issue as completed.",
        ]
    )


def format_signal_hits(title: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "- {}: none detected.".format(title)
    rows = []
    for hit in hits[:8]:
        label = (
            hit.get("nodeid")
            or hit.get("symbol")
            or hit.get("path")
            or hit.get("source")
            or hit.get("term")
            or "hit"
        )
        terms = ", ".join(hit.get("matched_terms", [])) or hit.get("term") or ""
        count = hit.get("count")
        details = terms or "matched"
        extra_details = []
        for field in ("kind", "category", "backend", "message"):
            if hit.get(field):
                extra_details.append("{}={}".format(field, hit[field]))
        if extra_details:
            details = "{}; {}".format(details, "; ".join(extra_details))
        if count is not None:
            details = "{}, count={}".format(details, count)
        rows.append("- {}: `{}` ({})".format(title, label, details))
    return "\n".join(rows)


def format_signal_section(signal: dict[str, Any] | None) -> str:
    if not signal:
        return "No generated support-signal data was provided for this row."
    lines = [
        "Extractor state: `{}`".format(signal.get("state", "unknown")),
        "Catalog evidence count: `{}`".format(signal.get("catalog_evidence_count", 0)),
        format_signal_hits("Tests", signal.get("tests", [])),
        format_signal_hits("Implementation", signal.get("implementation", [])),
        format_signal_hits("Unsupported markers", signal.get("unsupported", [])),
        format_signal_hits("Docs", signal.get("docs", [])),
        format_signal_hits("Pytest failures", signal.get("failures", [])),
    ]
    return "\n".join(lines)


def extracted_issue_body(
    issue: dict[str, Any],
    signals_lookup: dict[tuple[str, str], dict[str, Any]],
) -> str:
    signal = signals_lookup.get((issue["backend_id"], issue["feature_id"]))
    if signal is None:
        signal = issue.get("signal")
    return "\n\n".join(
        [
            marker_for(issue["key"]),
            "# {}: {}".format(issue["backend"], issue["feature"]),
            "This issue is managed from generated support signals by `tools/sync_support_issues.py`.",
            markdown_table(
                ["Field", "Value"],
                [
                    ["Backend", issue["backend"]],
                    ["Feature ID", issue["feature_id"]],
                    ["Category", issue["category"]],
                    ["Catalog status", issue["status"]],
                    ["Extractor state", issue["state"]],
                    ["Issue kind", issue["kind"]],
                ],
            ),
            "## Required Action\n\n{}".format(issue["title"]),
            "## Extracted Signals\n\n{}".format(format_signal_section(signal)),
            "## Completion Rule\n\nThe next sync closes this issue when generated extraction no longer reports this gap.",
        ]
    )


def build_desired_issues(
    matrix: dict[str, Any], signals: dict[str, Any] | None = None
) -> dict[str, DesiredIssue]:
    desired: dict[str, DesiredIssue] = {}
    lookup = feature_lookup(matrix)
    signals_by_row = signal_lookup(signals)
    rows_by_backend = backlog_by_backend(matrix)

    for backend in matrix.get("backends", []):
        backend_id = backend["id"]
        rows = rows_by_backend.get(backend_id, [])
        parent_key = "parent:{}".format(backend_id)
        parent_labels = (
            LABEL_MANAGED,
            LABEL_PARENT,
            LABEL_PREFIX_BACKEND + compact_label_part(backend_id),
        )
        desired[parent_key] = DesiredIssue(
            key=parent_key,
            title="[Support Matrix] {} coverage".format(backend["name"]),
            body=parent_body(matrix, backend, rows),
            labels=parent_labels,
        )

    frontend_key = "parent:{}".format(FRONTEND_ID)
    desired[frontend_key] = DesiredIssue(
        key=frontend_key,
        title="[Support Matrix] {} coverage".format(FRONTEND_NAME),
        body=frontend_parent_body(),
        labels=(
            LABEL_MANAGED,
            LABEL_PARENT,
            LABEL_PREFIX_BACKEND + FRONTEND_ID,
        ),
    )

    for item in matrix.get("backlog", []):
        backend_id = item["backend_id"]
        feature_id = item["feature_id"]
        key = "backlog:{}:{}".format(backend_id, feature_id)
        labels = (
            LABEL_MANAGED,
            LABEL_BACKLOG,
            LABEL_PREFIX_BACKEND + compact_label_part(backend_id),
            LABEL_PREFIX_CATEGORY + compact_label_part(item["category"]),
            LABEL_PREFIX_STATUS + compact_label_part(item["status"]),
        )
        desired[key] = DesiredIssue(
            key=key,
            title="[Support Matrix][{}] {} ({})".format(
                item["backend"], item["feature"], item["status"]
            ),
            body=child_body(item, lookup, signals_by_row),
            labels=labels,
            parent_key="parent:{}".format(backend_id),
        )

    for issue in (signals or {}).get("issues", []):
        key = issue["key"]
        if key in desired:
            continue
        labels = (
            LABEL_MANAGED,
            LABEL_EXTRACTED,
            LABEL_PREFIX_BACKEND + compact_label_part(issue["backend_id"]),
            LABEL_PREFIX_CATEGORY + compact_label_part(issue["category"]),
        )
        desired[key] = DesiredIssue(
            key=key,
            title="[Support Signals][{}] {} ({})".format(
                issue["backend"], issue["feature"], issue["kind"]
            ),
            body=extracted_issue_body(issue, signals_by_row),
            labels=labels,
            parent_key="parent:{}".format(issue["backend_id"]),
        )

    return desired


def desired_issue_counts(desired: dict[str, DesiredIssue]) -> dict[str, int]:
    return {
        "total": len(desired),
        "parents": sum(1 for key in desired if key.startswith("parent:")),
        "backlog": sum(1 for key in desired if key.startswith("backlog:")),
        "extracted": sum(1 for key in desired if key.startswith("extracted:")),
    }


def validate_desired_issues(
    matrix: dict[str, Any],
    signals: dict[str, Any] | None,
    desired: dict[str, DesiredIssue],
    *,
    min_desired_issues: int = 1,
) -> list[str]:
    errors: list[str] = []
    if len(desired) < min_desired_issues:
        errors.append(
            "desired issue plan has {} issues, below minimum {}".format(
                len(desired), min_desired_issues
            )
        )

    expected_parent_keys = {
        "parent:{}".format(backend["id"]) for backend in matrix.get("backends", [])
    }
    expected_parent_keys.add("parent:{}".format(FRONTEND_ID))
    for key in sorted(expected_parent_keys - set(desired)):
        errors.append("missing desired parent issue: {}".format(key))

    for item in matrix.get("backlog", []):
        if item.get("status") not in BACKLOG_STATUSES:
            errors.append(
                "backlog row {}:{} has non-backlog status {}".format(
                    item.get("backend_id"),
                    item.get("feature_id"),
                    item.get("status"),
                )
            )
        key = "backlog:{}:{}".format(item["backend_id"], item["feature_id"])
        if key not in desired:
            errors.append("missing desired backlog issue: {}".format(key))

    for issue in (signals or {}).get("issues", []):
        if issue["key"] not in desired:
            errors.append("missing desired extracted issue: {}".format(issue["key"]))

    for key, issue in desired.items():
        if marker_for(key) not in issue.body:
            errors.append(
                "desired issue {} body is missing its sync marker".format(key)
            )
        if LABEL_MANAGED not in issue.labels:
            errors.append("desired issue {} is missing managed label".format(key))
        if issue.parent_key and issue.parent_key not in desired:
            errors.append(
                "desired issue {} references missing parent {}".format(
                    key, issue.parent_key
                )
            )
        if key.startswith("backlog:") and LABEL_BACKLOG not in issue.labels:
            errors.append(
                "desired backlog issue {} is missing backlog label".format(key)
            )
        if key.startswith("extracted:") and LABEL_EXTRACTED not in issue.labels:
            errors.append(
                "desired extracted issue {} is missing extracted label".format(key)
            )

    return errors


def signals_allow_extracted_closure(signals: dict[str, Any] | None) -> bool:
    if not signals:
        return False
    docs_probe = signals.get("summary", {}).get("docs_probe", {})
    if not docs_probe.get("provided"):
        return False
    return int(docs_probe.get("failed", 0)) == 0


def is_pytest_failure_issue_key(key: str) -> bool:
    return (
        key.startswith("extracted:")
        and ":ci.pytest." in key
        and key.endswith(":pytest_failure_summary")
    )


def signals_allow_pytest_failure_closure(signals: dict[str, Any] | None) -> bool:
    if not signals:
        return False
    pytest_failures = signals.get("summary", {}).get("pytest_failures", {})
    return (
        bool(pytest_failures.get("provided"))
        and int(pytest_failures.get("load_error_count", 0)) == 0
    )


def stale_extracted_preserve_reason(
    key: str,
    *,
    close_extracted_issues: bool,
    close_pytest_failure_issues: bool,
) -> str | None:
    if key.startswith("extracted:") and not close_extracted_issues:
        return "stale_extracted_preserved"
    if is_pytest_failure_issue_key(key) and not close_pytest_failure_issues:
        return "stale_pytest_failure_preserved"
    return None


class GitHubClient:
    def __init__(
        self,
        repo: str,
        token: str,
        api_url: str = "https://api.github.com",
        max_retries: int = 4,
        retry_base_seconds: float = 30.0,
        retry_max_seconds: float = 300.0,
    ):
        if "/" not in repo:
            raise ValueError("Repository must be in OWNER/REPO form")
        self.repo = repo
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.max_retries = max(0, max_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.retry_max_seconds = max(self.retry_base_seconds, retry_max_seconds)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, str]]:
        for attempt in range(self.max_retries + 1):
            req = self.build_request(method, path, payload, query)
            try:
                with request.urlopen(req, timeout=30) as response:
                    text = response.read().decode("utf-8")
                    data = json.loads(text) if text else None
                    return data, {
                        key.lower(): value for key, value in response.headers.items()
                    }
            except error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                headers = {key.lower(): value for key, value in exc.headers.items()}
                if attempt < self.max_retries and self.should_retry_http_error(
                    exc.code, error_body
                ):
                    delay = self.retry_delay_seconds(headers, attempt)
                    print(
                        "GitHub API {} {} returned {}; retrying in {:.1f}s".format(
                            method, path, exc.code, delay
                        ),
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                raise GitHubApiError(
                    method, path, exc.code, error_body, headers
                ) from exc

        raise RuntimeError("unreachable GitHub request retry state")

    def build_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        query: dict[str, Any] | None,
    ) -> request.Request:
        url = self.api_url + path
        if query:
            url += "?" + parse.urlencode(query, doseq=True)
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("Authorization", "Bearer {}".format(self.token))
        req.add_header("X-GitHub-Api-Version", API_VERSION)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        return req

    def should_retry_http_error(self, status: int, body: str) -> bool:
        if status in {429, 500, 502, 503, 504}:
            return True
        return status == 403 and "secondary rate limit" in body.lower()

    def retry_delay_seconds(self, headers: dict[str, str], attempt: int) -> float:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass

        if headers.get("x-ratelimit-remaining") == "0":
            reset_at = headers.get("x-ratelimit-reset")
            if reset_at:
                try:
                    return max(0.0, float(reset_at) - time.time() + 5.0)
                except ValueError:
                    pass

        delay = self.retry_base_seconds * (2**attempt)
        return min(self.retry_max_seconds, delay)

    def paginate(
        self, path: str, query: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        query = dict(query or {})
        query.setdefault("per_page", 100)
        items: list[dict[str, Any]] = []
        next_url = self.api_url + path + "?" + parse.urlencode(query, doseq=True)
        while next_url:
            parsed = parse.urlparse(next_url)
            data, headers = self.request(
                "GET",
                parsed.path,
                query=dict(parse.parse_qsl(parsed.query)),
            )
            items.extend(data or [])
            next_url = parse_link_header(headers.get("link", "")).get("next")
        return items

    def list_managed_issues(self) -> list[dict[str, Any]]:
        owner_repo = "/repos/{}".format(self.repo)
        return self.paginate(
            owner_repo + "/issues",
            {
                "state": "all",
                "labels": LABEL_MANAGED,
            },
        )

    def ensure_label(self, name: str, color: str, description: str) -> None:
        path = "/repos/{}/labels".format(self.repo)
        try:
            self.request(
                "POST",
                path,
                {
                    "name": name,
                    "color": color,
                    "description": description[:100],
                },
            )
        except GitHubApiError as exc:
            if exc.status != 422:
                raise
            encoded_name = parse.quote(name, safe="")
            self.request(
                "PATCH",
                path + "/" + encoded_name,
                {
                    "new_name": name,
                    "color": color,
                    "description": description[:100],
                },
            )

    def create_issue(self, desired: DesiredIssue) -> dict[str, Any]:
        issue, _ = self.request(
            "POST",
            "/repos/{}/issues".format(self.repo),
            {
                "title": desired.title,
                "body": desired.body,
                "labels": list(desired.labels),
            },
        )
        return issue

    def update_issue(
        self, issue: dict[str, Any], desired: DesiredIssue
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if issue.get("title") != desired.title:
            payload["title"] = desired.title
        if issue.get("body") != desired.body:
            payload["body"] = desired.body
        if issue.get("state") != "open":
            payload["state"] = "open"
        current_labels = {label["name"] for label in issue.get("labels", [])}
        if current_labels != set(desired.labels):
            payload["labels"] = list(desired.labels)
        if not payload:
            return issue
        updated, _ = self.request(
            "PATCH",
            "/repos/{}/issues/{}".format(self.repo, issue["number"]),
            payload,
        )
        return updated

    def close_issue(self, issue: dict[str, Any], body: str) -> dict[str, Any]:
        payload = {
            "state": "closed",
            "state_reason": "completed",
            "body": body,
        }
        updated, _ = self.request(
            "PATCH",
            "/repos/{}/issues/{}".format(self.repo, issue["number"]),
            payload,
        )
        return updated

    def add_sub_issue(self, parent: dict[str, Any], child: dict[str, Any]) -> None:
        try:
            self.request(
                "POST",
                "/repos/{}/issues/{}/sub_issues".format(self.repo, parent["number"]),
                {
                    "sub_issue_id": child["id"],
                    "replace_parent": True,
                },
            )
        except GitHubApiError as exc:
            # 422 is returned both for duplicates and validation failures. With
            # replace_parent=true, a duplicate relationship is already the desired
            # state, so treat it as non-fatal and let true validation failures
            # surface through issue body/state mismatches on the next sync.
            if exc.status != 422:
                raise

    def list_sub_issues(self, parent: dict[str, Any]) -> list[dict[str, Any]]:
        return self.paginate(
            "/repos/{}/issues/{}/sub_issues".format(self.repo, parent["number"])
        )


def parse_link_header(value: str) -> dict[str, str]:
    links = {}
    for part in value.split(","):
        section = part.strip()
        if not section:
            continue
        match = re.match(r"<([^>]+)>;\s*rel=\"([^\"]+)\"", section)
        if match:
            links[match.group(2)] = match.group(1)
    return links


def desired_label_catalog(
    desired: dict[str, DesiredIssue],
) -> dict[str, tuple[str, str]]:
    labels = {
        LABEL_MANAGED: ("5319e7", "Managed from support/generated/support-matrix.json"),
        LABEL_PARENT: ("0e8a16", "Parent issue for support matrix backlog"),
        LABEL_BACKLOG: ("fbca04", "Support matrix backlog sub-issue"),
        LABEL_EXTRACTED: ("fef2c0", "Generated support extraction issue"),
    }
    for issue in desired.values():
        for label in issue.labels:
            if label in labels:
                continue
            if label.startswith(LABEL_PREFIX_STATUS):
                labels[label] = ("d4c5f9", "Support matrix status")
            elif label.startswith(LABEL_PREFIX_BACKEND):
                labels[label] = ("bfd4f2", "Support matrix backend or component")
            elif label.startswith(LABEL_PREFIX_CATEGORY):
                labels[label] = ("c2e0c6", "Support matrix feature category")
            else:
                labels[label] = ("eeeeee", "Support matrix issue label")
    return labels


def closed_body(issue: dict[str, Any], key: str) -> str:
    return "\n\n".join(
        [
            marker_for(key),
            "# Completed Managed Support Issue",
            "This managed issue was closed because the corresponding support matrix or generated support-signal row is no longer present.",
            "Previous title: `{}`".format(issue.get("title", "")),
        ]
    )


def duplicate_closed_body(issue: dict[str, Any], key: str) -> str:
    return "\n\n".join(
        [
            marker_for(key),
            "# Duplicate Managed Support Issue",
            "This managed issue was closed because another managed issue already owns the same support sync marker.",
            "Sync marker: `{}`".format(key),
            "Previous title: `{}`".format(issue.get("title", "")),
        ]
    )


def issue_labels(issue: dict[str, Any]) -> set[str]:
    return {label["name"] for label in issue.get("labels", [])}


def empty_sync_summary() -> dict[str, int]:
    return {
        "created": 0,
        "updated": 0,
        "closed": 0,
        "attached": 0,
        "unchanged": 0,
    }


def split_existing_issues(
    existing_issues: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, dict[str, Any]]], int]:
    existing_by_key: dict[str, dict[str, Any]] = {}
    duplicate_existing: list[tuple[str, dict[str, Any]]] = []
    unmarked = 0
    for issue in existing_issues:
        key = marker_key(issue.get("body"))
        if not key:
            unmarked += 1
            continue
        current = existing_by_key.get(key)
        if current is None:
            existing_by_key[key] = issue
        elif current.get("state") == "closed" and issue.get("state") != "closed":
            duplicate_existing.append((key, current))
            existing_by_key[key] = issue
        else:
            duplicate_existing.append((key, issue))
    return existing_by_key, duplicate_existing, unmarked


def issue_requires_update(issue: dict[str, Any], desired: DesiredIssue) -> bool:
    return (
        issue.get("title") != desired.title
        or issue.get("body") != desired.body
        or issue.get("state") != "open"
        or issue_labels(issue) != set(desired.labels)
    )


def issue_update_reasons(issue: dict[str, Any], desired: DesiredIssue) -> list[str]:
    reasons = []
    if issue.get("title") != desired.title:
        reasons.append("title")
    if issue.get("body") != desired.body:
        reasons.append("body")
    if issue.get("state") != "open":
        reasons.append("state")
    if issue_labels(issue) != set(desired.labels):
        reasons.append("labels")
    return reasons


def issue_reference(issue: dict[str, Any], key: str | None = None) -> dict[str, Any]:
    reference = {
        "key": key or marker_key(issue.get("body")),
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "state": issue.get("state", "unknown"),
    }
    if issue.get("html_url"):
        reference["url"] = issue["html_url"]
    return reference


def issue_operation_reference(
    key: str | None = None,
    issue: dict[str, Any] | None = None,
    desired: DesiredIssue | None = None,
) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    if key is not None:
        reference["key"] = key
    if issue is not None:
        reference.update(
            {
                "number": issue.get("number"),
                "title": issue.get("title", ""),
                "state": issue.get("state", "unknown"),
            }
        )
    if desired is not None:
        reference.setdefault("key", desired.key)
        reference["title"] = desired.title
        if desired.parent_key:
            reference["parent_key"] = desired.parent_key
    return reference


def issue_operation_ledger_entry(
    action: str,
    key: str | None = None,
    issue: dict[str, Any] | None = None,
    desired: DesiredIssue | None = None,
    **extra: Any,
) -> dict[str, Any]:
    entry = {
        "action": action,
        **issue_operation_reference(key, issue, desired),
    }
    entry.update({name: value for name, value in extra.items() if value is not None})
    return entry


def desired_issue_reference(key: str, desired: DesiredIssue) -> dict[str, Any]:
    reference = {
        "key": key,
        "title": desired.title,
    }
    if desired.parent_key:
        reference["parent_key"] = desired.parent_key
    return reference


def append_planned_action_sample(
    samples: dict[str, Any],
    action: str,
    sample: dict[str, Any],
    sample_limit: int,
) -> None:
    if len(samples[action]) < sample_limit:
        samples[action].append(sample)


def empty_planned_action_samples(sample_limit: int) -> dict[str, Any]:
    return {
        "sample_limit": sample_limit,
        "created": [],
        "updated": [],
        "closed": [],
        "attached": [],
        "preserved": [],
    }


def planned_issue_action_samples(
    desired: dict[str, DesiredIssue],
    existing_issues: list[dict[str, Any]],
    *,
    manage_sub_issues: bool = True,
    close_extracted_issues: bool = True,
    close_pytest_failure_issues: bool = True,
    existing_sub_issue_ids_by_parent: dict[int, set[int]] | None = None,
    sample_limit: int = PLANNED_ACTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    samples = empty_planned_action_samples(sample_limit)
    existing_sub_issue_ids_by_parent = existing_sub_issue_ids_by_parent or {}
    existing_by_key, duplicate_existing, _unmarked = split_existing_issues(
        existing_issues
    )

    for key, target in desired.items():
        issue = existing_by_key.get(key)
        if issue is None:
            append_planned_action_sample(
                samples,
                "created",
                desired_issue_reference(key, target),
                sample_limit,
            )
        elif issue_requires_update(issue, target):
            sample = issue_reference(issue, key)
            sample["reasons"] = issue_update_reasons(issue, target)
            append_planned_action_sample(samples, "updated", sample, sample_limit)

    if manage_sub_issues:
        for key, target in desired.items():
            if not target.parent_key:
                continue
            parent = existing_by_key.get(target.parent_key)
            child = existing_by_key.get(key)
            if parent is not None and child is not None:
                existing_child_ids = existing_sub_issue_ids_by_parent.get(
                    parent["number"], set()
                )
                if child.get("id") in existing_child_ids:
                    continue
                reason = "missing_relationship"
            else:
                reason = "parent_or_child_will_be_created"
            append_planned_action_sample(
                samples,
                "attached",
                {
                    "parent_key": target.parent_key,
                    "child_key": key,
                    "reason": reason,
                },
                sample_limit,
            )

    for key, issue in existing_by_key.items():
        if key in desired:
            continue
        preserve_reason = stale_extracted_preserve_reason(
            key,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
        )
        if preserve_reason is not None:
            sample = issue_reference(issue, key)
            sample["reason"] = preserve_reason
            append_planned_action_sample(samples, "preserved", sample, sample_limit)
            continue
        if not key.startswith(("backlog:", "parent:", "extracted:")):
            continue
        if issue.get("state") == "closed":
            continue
        sample = issue_reference(issue, key)
        sample["reason"] = "stale_managed_marker"
        append_planned_action_sample(samples, "closed", sample, sample_limit)

    for key, issue in duplicate_existing:
        if issue.get("state") == "closed":
            continue
        sample = issue_reference(issue, key)
        sample["reason"] = "duplicate_managed_marker"
        append_planned_action_sample(samples, "closed", sample, sample_limit)

    return samples


def planned_issue_actions(
    desired: dict[str, DesiredIssue],
    existing_issues: list[dict[str, Any]],
    *,
    manage_sub_issues: bool = True,
    close_extracted_issues: bool = True,
    close_pytest_failure_issues: bool = True,
    existing_sub_issue_ids_by_parent: dict[int, set[int]] | None = None,
) -> dict[str, int]:
    summary = empty_sync_summary()
    existing_sub_issue_ids_by_parent = existing_sub_issue_ids_by_parent or {}
    existing_by_key, duplicate_existing, _unmarked = split_existing_issues(
        existing_issues
    )

    for key, target in desired.items():
        issue = existing_by_key.get(key)
        if issue is None:
            summary["created"] += 1
        elif issue_requires_update(issue, target):
            summary["updated"] += 1
        else:
            summary["unchanged"] += 1

    if manage_sub_issues:
        for key, target in desired.items():
            if not target.parent_key:
                continue
            parent = existing_by_key.get(target.parent_key)
            child = existing_by_key.get(key)
            if parent is not None and child is not None:
                existing_child_ids = existing_sub_issue_ids_by_parent.get(
                    parent["number"], set()
                )
                if child.get("id") in existing_child_ids:
                    continue
            summary["attached"] += 1

    for key, issue in existing_by_key.items():
        if key in desired:
            continue
        preserve_reason = stale_extracted_preserve_reason(
            key,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
        )
        if preserve_reason is not None:
            summary["unchanged"] += 1
            continue
        if not key.startswith(("backlog:", "parent:", "extracted:")):
            continue
        if issue.get("state") == "closed":
            summary["unchanged"] += 1
            continue
        summary["closed"] += 1

    for _key, issue in duplicate_existing:
        if issue.get("state") == "closed":
            summary["unchanged"] += 1
            continue
        summary["closed"] += 1

    return summary


def empty_closure_summary() -> dict[str, int]:
    return {
        "total": 0,
        "stale_parent": 0,
        "stale_backlog": 0,
        "stale_extracted": 0,
        "duplicate_marker": 0,
    }


def stale_closure_category(key: str) -> str | None:
    if key.startswith("parent:"):
        return "stale_parent"
    if key.startswith("backlog:"):
        return "stale_backlog"
    if key.startswith("extracted:"):
        return "stale_extracted"
    return None


def planned_issue_closures(
    desired: dict[str, DesiredIssue],
    existing_issues: list[dict[str, Any]],
    *,
    close_extracted_issues: bool = True,
    close_pytest_failure_issues: bool = True,
) -> dict[str, int]:
    summary = empty_closure_summary()
    existing_by_key, duplicate_existing, _unmarked = split_existing_issues(
        existing_issues
    )

    for key, issue in existing_by_key.items():
        if key in desired:
            continue
        preserve_reason = stale_extracted_preserve_reason(
            key,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
        )
        if preserve_reason is not None:
            continue
        if issue.get("state") == "closed":
            continue
        category = stale_closure_category(key)
        if category is None:
            continue
        summary[category] += 1
        summary["total"] += 1

    for _key, issue in duplicate_existing:
        if issue.get("state") == "closed":
            continue
        summary["duplicate_marker"] += 1
        summary["total"] += 1

    return summary


def append_audit_sample(
    bucket: dict[str, Any],
    sample: dict[str, Any],
    sample_limit: int,
) -> None:
    if len(bucket["samples"]) < sample_limit:
        bucket["samples"].append(sample)


def audit_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "open": 0,
        "closed": 0,
        "samples": [],
    }


def managed_issue_audit(
    desired: dict[str, DesiredIssue],
    existing_issues: list[dict[str, Any]],
    *,
    close_extracted_issues: bool = True,
    close_pytest_failure_issues: bool = True,
    sample_limit: int = PLANNED_ACTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    existing_by_key, duplicate_existing, _unmarked = split_existing_issues(
        existing_issues
    )
    audit = {
        "sample_limit": sample_limit,
        "stale": audit_bucket(),
        "duplicates": audit_bucket(),
        "preserved_extracted": audit_bucket(),
        "ignored_unknown": audit_bucket(),
    }

    for key, issue in existing_by_key.items():
        if key in desired:
            continue

        preserve_reason = stale_extracted_preserve_reason(
            key,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
        )
        if preserve_reason is not None:
            bucket = audit["preserved_extracted"]
            bucket["total"] += 1
            if issue.get("state") == "closed":
                bucket["closed"] += 1
            else:
                bucket["open"] += 1
            sample = issue_reference(issue, key)
            sample["reason"] = preserve_reason
            append_audit_sample(bucket, sample, sample_limit)
            continue

        category = stale_closure_category(key)
        if category is None:
            bucket = audit["ignored_unknown"]
            bucket["total"] += 1
            if issue.get("state") == "closed":
                bucket["closed"] += 1
            else:
                bucket["open"] += 1
            sample = issue_reference(issue, key)
            sample["reason"] = "unknown_managed_marker"
            append_audit_sample(bucket, sample, sample_limit)
            continue

        bucket = audit["stale"]
        bucket["total"] += 1
        if issue.get("state") == "closed":
            bucket["closed"] += 1
            reason = "closed_stale_managed_marker"
        else:
            bucket["open"] += 1
            reason = "stale_managed_marker"
        sample = issue_reference(issue, key)
        sample["category"] = category
        sample["reason"] = reason
        append_audit_sample(bucket, sample, sample_limit)

    for key, issue in duplicate_existing:
        bucket = audit["duplicates"]
        bucket["total"] += 1
        if issue.get("state") == "closed":
            bucket["closed"] += 1
            reason = "closed_duplicate_managed_marker"
        else:
            bucket["open"] += 1
            reason = "duplicate_managed_marker"
        sample = issue_reference(issue, key)
        sample["reason"] = reason
        append_audit_sample(bucket, sample, sample_limit)

    return audit


def planned_action_budget_report(
    planned_actions: dict[str, int] | None,
    limits: dict[str, int],
    *,
    mode: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "provided": bool(limits),
        "mode": mode,
        "evaluated": planned_actions is not None,
        "ok": None,
        "limits": limits,
        "violations": [],
    }
    if not limits:
        return report
    if planned_actions is None:
        return report

    comparable_actions = dict(planned_actions)
    comparable_actions["total"] = sum(
        planned_actions.get(action, 0)
        for action in ("created", "updated", "closed", "attached")
    )
    violations = []
    for action, limit in sorted(limits.items()):
        actual = comparable_actions.get(action, 0)
        if actual > limit:
            violations.append(
                {
                    "action": action,
                    "actual": actual,
                    "limit": limit,
                }
            )

    report["ok"] = not violations
    report["violations"] = violations
    return report


def planned_closure_budget_report(
    planned_closures: dict[str, int] | None,
    limits: dict[str, int],
    *,
    mode: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "provided": bool(limits),
        "mode": mode,
        "evaluated": planned_closures is not None,
        "ok": None,
        "limits": limits,
        "violations": [],
    }
    if not limits:
        return report
    if planned_closures is None:
        return report

    violations = []
    for category, limit in sorted(limits.items()):
        actual = planned_closures.get(category, 0)
        if actual > limit:
            violations.append(
                {
                    "category": category,
                    "actual": actual,
                    "limit": limit,
                }
            )

    report["ok"] = not violations
    report["violations"] = violations
    return report


def planned_closure_budget_errors(report: dict[str, Any]) -> list[str]:
    if not report.get("provided") or not report.get("evaluated"):
        return []
    return [
        (
            "planned issue closure budget exceeded for "
            "{category}: {actual} > {limit}"
        ).format(**violation)
        for violation in report.get("violations", [])
    ]


def planned_action_budget_errors(report: dict[str, Any]) -> list[str]:
    if not report.get("provided") or not report.get("evaluated"):
        return []
    return [
        "planned issue action budget exceeded for {action}: {actual} > {limit}".format(
            **violation
        )
        for violation in report.get("violations", [])
    ]


def operation_ledger_action_counts(
    operation_ledger: list[dict[str, Any]] | None,
) -> dict[str, int]:
    counts = {action: 0 for action in ("created", "updated", "closed", "attached")}
    for entry in operation_ledger or []:
        action = entry.get("action")
        if action in counts:
            counts[action] += 1
    return counts


def operation_ledger_closure_counts(
    operation_ledger: list[dict[str, Any]] | None,
) -> dict[str, int]:
    counts = empty_closure_summary()
    for entry in operation_ledger or []:
        if entry.get("action") != "closed":
            continue
        if entry.get("reason") == "duplicate_managed_marker":
            category = "duplicate_marker"
        else:
            category = stale_closure_category(str(entry.get("key", "")))
        if category is None:
            continue
        counts[category] += 1
        counts["total"] += 1
    return counts


def operation_reconciliation_report(
    planned_actions: dict[str, int] | None,
    planned_closures: dict[str, int] | None,
    operation_ledger: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "evaluated": planned_actions is not None and operation_ledger is not None,
        "ok": None,
        "planned_actions": planned_actions,
        "actual_actions": None,
        "action_overruns": [],
        "planned_closures": planned_closures,
        "actual_closures": None,
        "closure_overruns": [],
    }
    if planned_actions is None or operation_ledger is None:
        return report

    actual_actions = operation_ledger_action_counts(operation_ledger)
    action_overruns = []
    for action in sorted(actual_actions):
        actual = actual_actions[action]
        planned = planned_actions.get(action, 0)
        if actual > planned:
            action_overruns.append(
                {
                    "action": action,
                    "actual": actual,
                    "planned": planned,
                }
            )

    actual_closures = operation_ledger_closure_counts(operation_ledger)
    closure_overruns = []
    if planned_closures is not None:
        for category in sorted(actual_closures):
            actual = actual_closures[category]
            planned = planned_closures.get(category, 0)
            if actual > planned:
                closure_overruns.append(
                    {
                        "category": category,
                        "actual": actual,
                        "planned": planned,
                    }
                )

    report.update(
        {
            "ok": not action_overruns and not closure_overruns,
            "actual_actions": actual_actions,
            "action_overruns": action_overruns,
            "actual_closures": actual_closures,
            "closure_overruns": closure_overruns,
        }
    )
    return report


def inspect_existing_issue_state(
    client: GitHubClient,
    desired: dict[str, DesiredIssue],
    *,
    manage_sub_issues: bool = True,
) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
    try:
        existing_issues = client.list_managed_issues()
    except Exception as exc:
        raise_sync_preflight_error("list_managed_issues", {}, exc)
    existing_sub_issue_ids_by_parent: dict[int, set[int]] = {}
    if manage_sub_issues:
        existing_by_key, _duplicates, _unmarked = split_existing_issues(existing_issues)
        for target in desired.values():
            if target.parent_key:
                continue
            parent = existing_by_key.get(target.key)
            if parent is None:
                continue
            try:
                existing_sub_issue_ids_by_parent[parent["number"]] = {
                    item["id"] for item in client.list_sub_issues(parent)
                }
            except Exception as exc:
                raise_sync_preflight_error(
                    "list_sub_issues",
                    {
                        "parent_key": target.key,
                        "parent_number": parent["number"],
                    },
                    exc,
                )
    return existing_issues, existing_sub_issue_ids_by_parent


def issue_sync_report(
    desired: dict[str, DesiredIssue],
    *,
    mode: str,
    close_extracted_issues: bool,
    close_pytest_failure_issues: bool = True,
    manage_sub_issues: bool,
    matrix_check_report: dict[str, Any] | None = None,
    matrix_check_report_path: Path | None = None,
    planned_action_budget_limits: dict[str, int] | None = None,
    planned_action_budget_mode: str = "fail",
    existing_issues: list[dict[str, Any]] | None = None,
    existing_sub_issue_ids_by_parent: dict[int, set[int]] | None = None,
    sync_summary: dict[str, int] | None = None,
    planned_closure_budget_limits: dict[str, int] | None = None,
    preflight_failure: dict[str, Any] | None = None,
    sync_failure: dict[str, Any] | None = None,
    input_failures: list[dict[str, Any]] | None = None,
    operation_ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    planned_actions = None
    planned_closures = None
    if existing_issues is not None:
        planned_actions = planned_issue_actions(
            desired,
            existing_issues,
            manage_sub_issues=manage_sub_issues,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
            existing_sub_issue_ids_by_parent=existing_sub_issue_ids_by_parent,
        )
        planned_closures = planned_issue_closures(
            desired,
            existing_issues,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "generator": "tools/sync_support_issues.py",
        "mode": mode,
        "desired": desired_issue_counts(desired),
        "close_extracted_issues": close_extracted_issues,
        "close_pytest_failure_issues": close_pytest_failure_issues,
        "manage_sub_issues": manage_sub_issues,
        "support_matrix_check": support_matrix_check_summary(
            matrix_check_report,
            matrix_check_report_path,
        ),
        "planned_action_budget": planned_action_budget_report(
            planned_actions,
            planned_action_budget_limits or {},
            mode=planned_action_budget_mode,
        ),
        "planned_closure_budget": planned_closure_budget_report(
            planned_closures,
            planned_closure_budget_limits or {},
            mode=planned_action_budget_mode,
        ),
        "existing": {
            "inspected": existing_issues is not None,
            "managed": 0,
            "duplicates": 0,
            "unmarked": 0,
        },
        "input_failures": input_failures or [],
        "operation_ledger": operation_ledger,
        "planned_actions": planned_actions,
        "planned_closures": planned_closures,
        "operation_reconciliation": operation_reconciliation_report(
            planned_actions,
            planned_closures,
            operation_ledger,
        ),
        "planned_action_samples": (
            planned_issue_action_samples(
                desired,
                existing_issues,
                manage_sub_issues=manage_sub_issues,
                close_extracted_issues=close_extracted_issues,
                close_pytest_failure_issues=close_pytest_failure_issues,
                existing_sub_issue_ids_by_parent=existing_sub_issue_ids_by_parent,
            )
            if existing_issues is not None
            else None
        ),
        "managed_issue_audit": (
            managed_issue_audit(
                desired,
                existing_issues,
                close_extracted_issues=close_extracted_issues,
                close_pytest_failure_issues=close_pytest_failure_issues,
            )
            if existing_issues is not None
            else None
        ),
    }
    if existing_issues is not None:
        existing_by_key, duplicates, unmarked = split_existing_issues(existing_issues)
        report["existing"] = {
            "inspected": True,
            "managed": len(existing_by_key),
            "duplicates": len(duplicates),
            "unmarked": unmarked,
        }
    if sync_summary is not None:
        report["sync_summary"] = sync_summary
    if preflight_failure is not None:
        report["preflight_failure"] = preflight_failure
    if sync_failure is not None:
        report["sync_failure"] = sync_failure
    return report


def sync_exception_summary(cause: Exception) -> dict[str, Any]:
    error_summary: dict[str, Any] = {
        "type": type(cause).__name__,
        "message": str(cause),
    }
    if isinstance(cause, GitHubApiError):
        error_summary.update(
            {
                "method": cause.method,
                "path": cause.path,
                "status": cause.status,
                "body": cause.body[:1000],
            }
        )
    return error_summary


def sync_failure_summary(exc: SupportIssueSyncMutationError) -> dict[str, Any]:
    return {
        "phase": exc.phase,
        "operation": exc.operation,
        "partial_summary": dict(exc.summary),
        "operation_ledger": list(exc.operation_ledger),
        "error": sync_exception_summary(exc.cause),
        "recovery": {
            "rerun_safe": True,
            "strategy": (
                "Rerun support issue sync after correcting the failure; "
                "managed issue markers make completed create, update, close, "
                "and attach operations idempotent."
            ),
        },
    }


def preflight_failure_summary(exc: SupportIssueSyncPreflightError) -> dict[str, Any]:
    return {
        "phase": exc.phase,
        "operation": exc.operation,
        "error": sync_exception_summary(exc.cause),
    }


def support_matrix_check_summary(
    report: dict[str, Any] | None,
    path: Path | None = None,
) -> dict[str, Any]:
    source_path = str(path) if path is not None else None
    if not report:
        return {
            "provided": False,
            "path": source_path,
        }
    load_error = support_matrix_check_report_error(report, path)
    if load_error is not None:
        return {
            "provided": True,
            "path": source_path,
            "ok": False,
            "summary": {},
            "stale_artifacts": [],
            "load_error": load_error,
        }

    artifacts = report.get("artifacts", []) or []
    stale_artifacts = [
        {
            "path": artifact.get("path"),
            "diff_line_count": artifact.get("diff_line_count", 0),
            "actual_sha256": artifact.get("actual_sha256"),
            "expected_sha256": artifact.get("expected_sha256"),
        }
        for artifact in artifacts
        if artifact.get("stale")
    ]
    return {
        "provided": True,
        "path": source_path,
        "ok": bool(report.get("ok")),
        "summary": report.get("summary", {}),
        "stale_artifacts": stale_artifacts,
    }


def support_matrix_check_report_error(
    report: dict[str, Any],
    path: Path | None = None,
) -> dict[str, Any] | None:
    if "load_error" in report:
        return report["load_error"]

    schema_error = json_report_schema_error(
        report,
        path,
        expected_generator=MATRIX_CHECK_REPORT_GENERATOR,
        required_fields=MATRIX_CHECK_REPORT_REQUIRED_FIELDS,
        schema_version=MATRIX_CHECK_REPORT_SCHEMA_VERSION,
    )
    if schema_error is not None:
        return schema_error

    if not isinstance(report.get("summary"), dict):
        return optional_json_load_error(
            path,
            "InvalidReportFieldType",
            "expected summary to be object, got {}".format(
                type(report.get("summary")).__name__
            ),
        )["load_error"]

    if not isinstance(report.get("artifacts"), list):
        return optional_json_load_error(
            path,
            "InvalidReportFieldType",
            "expected artifacts to be list, got {}".format(
                type(report.get("artifacts")).__name__
            ),
        )["load_error"]

    return None


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Wrote {}".format(path))


def raise_sync_mutation_error(
    phase: str,
    operation: dict[str, Any],
    summary: dict[str, int],
    cause: Exception,
    operation_ledger: list[dict[str, Any]] | None = None,
) -> None:
    if isinstance(cause, SupportIssueSyncMutationError):
        raise cause
    raise SupportIssueSyncMutationError(
        phase,
        operation,
        summary,
        cause,
        operation_ledger=operation_ledger,
    ) from cause


def raise_sync_preflight_error(
    phase: str,
    operation: dict[str, Any],
    cause: Exception,
) -> None:
    if isinstance(cause, SupportIssueSyncPreflightError):
        raise cause
    raise SupportIssueSyncPreflightError(phase, operation, cause) from cause


def sync_issues(
    client: GitHubClient,
    desired: dict[str, DesiredIssue],
    *,
    dry_run: bool = False,
    manage_sub_issues: bool = True,
    close_extracted_issues: bool = True,
    close_pytest_failure_issues: bool = True,
    throttle_seconds: float = 0.2,
    operation_ledger: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    summary = empty_sync_summary()
    ledger = operation_ledger if operation_ledger is not None else []
    try:
        existing_issues = client.list_managed_issues() if not dry_run else []
    except Exception as exc:
        raise_sync_mutation_error(
            "list_managed_issues", {}, summary, exc, operation_ledger=ledger
        )
    existing_by_key, duplicate_existing, _unmarked = split_existing_issues(
        existing_issues
    )

    if dry_run:
        print("Dry run: would manage {} desired issues".format(len(desired)))
        parent_count = sum(1 for key in desired if key.startswith("parent:"))
        child_count = sum(1 for key in desired if key.startswith("backlog:"))
        extracted_count = sum(1 for key in desired if key.startswith("extracted:"))
        print("Parents: {}".format(parent_count))
        print("Backlog sub-issues: {}".format(child_count))
        print("Extracted signal sub-issues: {}".format(extracted_count))
        return summary

    for label, (color, description) in desired_label_catalog(desired).items():
        try:
            client.ensure_label(label, color, description)
        except Exception as exc:
            raise_sync_mutation_error(
                "ensure_label",
                {"label": label},
                summary,
                exc,
                operation_ledger=ledger,
            )
        time.sleep(throttle_seconds)

    materialized: dict[str, dict[str, Any]] = {}
    for key, target in desired.items():
        issue = existing_by_key.get(key)
        if issue is None:
            try:
                issue = client.create_issue(target)
            except Exception as exc:
                raise_sync_mutation_error(
                    "create_issue",
                    issue_operation_reference(key, desired=target),
                    summary,
                    exc,
                    operation_ledger=ledger,
                )
            summary["created"] += 1
            ledger.append(issue_operation_ledger_entry("created", key, issue, target))
            time.sleep(throttle_seconds)
        else:
            update_reasons = issue_update_reasons(issue, target)
            before = (
                issue.get("title"),
                issue.get("body"),
                issue.get("state"),
                issue_labels(issue),
            )
            try:
                issue = client.update_issue(issue, target)
            except Exception as exc:
                raise_sync_mutation_error(
                    "update_issue",
                    issue_operation_reference(key, issue, target),
                    summary,
                    exc,
                    operation_ledger=ledger,
                )
            after = (
                issue.get("title"),
                issue.get("body"),
                issue.get("state"),
                issue_labels(issue),
            )
            if before == after:
                summary["unchanged"] += 1
            else:
                summary["updated"] += 1
                ledger.append(
                    issue_operation_ledger_entry(
                        "updated",
                        key,
                        issue,
                        target,
                        reasons=update_reasons,
                    )
                )
                time.sleep(throttle_seconds)
        materialized[key] = issue

    if manage_sub_issues:
        child_ids_by_parent: dict[int, set[int]] = {}
        for key, target in desired.items():
            if not target.parent_key:
                continue
            parent = materialized.get(target.parent_key)
            child = materialized.get(key)
            if parent is None or child is None:
                continue
            parent_number = parent["number"]
            if parent_number not in child_ids_by_parent:
                try:
                    child_ids_by_parent[parent_number] = {
                        item["id"] for item in client.list_sub_issues(parent)
                    }
                except Exception as exc:
                    raise_sync_mutation_error(
                        "list_sub_issues",
                        {
                            "parent_key": target.parent_key,
                            "parent_number": parent_number,
                        },
                        summary,
                        exc,
                        operation_ledger=ledger,
                    )
                time.sleep(throttle_seconds)
            if child["id"] in child_ids_by_parent[parent_number]:
                continue
            try:
                client.add_sub_issue(parent, child)
            except Exception as exc:
                raise_sync_mutation_error(
                    "add_sub_issue",
                    {
                        "parent_key": target.parent_key,
                        "parent_number": parent_number,
                        "child_key": key,
                        "child_number": child.get("number"),
                    },
                    summary,
                    exc,
                    operation_ledger=ledger,
                )
            child_ids_by_parent[parent_number].add(child["id"])
            summary["attached"] += 1
            ledger.append(
                {
                    "action": "attached",
                    "parent_key": target.parent_key,
                    "parent_number": parent_number,
                    "child_key": key,
                    "child_number": child.get("number"),
                }
            )
            time.sleep(throttle_seconds)

    for key, issue in existing_by_key.items():
        if key in desired:
            continue
        preserve_reason = stale_extracted_preserve_reason(
            key,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
        )
        if preserve_reason is not None:
            summary["unchanged"] += 1
            continue
        if not key.startswith(("backlog:", "parent:", "extracted:")):
            continue
        if issue.get("state") == "closed":
            summary["unchanged"] += 1
            continue
        try:
            client.close_issue(issue, closed_body(issue, key))
        except Exception as exc:
            raise_sync_mutation_error(
                "close_stale_issue",
                issue_operation_reference(key, issue),
                summary,
                exc,
                operation_ledger=ledger,
            )
        summary["closed"] += 1
        ledger.append(
            issue_operation_ledger_entry(
                "closed",
                key,
                issue,
                reason="stale_managed_marker",
            )
        )
        time.sleep(throttle_seconds)

    for key, issue in duplicate_existing:
        if issue.get("state") == "closed":
            summary["unchanged"] += 1
            continue
        try:
            client.close_issue(issue, duplicate_closed_body(issue, key))
        except Exception as exc:
            raise_sync_mutation_error(
                "close_duplicate_issue",
                issue_operation_reference(key, issue),
                summary,
                exc,
                operation_ledger=ledger,
            )
        summary["closed"] += 1
        ledger.append(
            issue_operation_ledger_entry(
                "closed",
                key,
                issue,
                reason="duplicate_managed_marker",
            )
        )
        time.sleep(throttle_seconds)

    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="Path to generated support-matrix JSON",
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=DEFAULT_SIGNALS_PATH,
        help="Optional generated support-signals JSON path",
    )
    parser.add_argument(
        "--matrix-check-report",
        type=Path,
        default=DEFAULT_MATRIX_CHECK_REPORT_PATH,
        help=(
            "Optional support_matrix.py check JSON report path to summarize in "
            "issue-sync plan outputs"
        ),
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository in OWNER/REPO form",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing the GitHub token",
    )
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--inspect-existing",
        action="store_true",
        help="Read existing managed issues and include planned actions in the JSON plan",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="Optional JSON path for the pre-mutation issue plan",
    )
    parser.add_argument(
        "--sync-summary-output",
        type=Path,
        help="Optional JSON path for the post-sync action summary",
    )
    parser.add_argument("--no-sub-issues", action="store_true")
    parser.add_argument("--throttle-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=30.0)
    parser.add_argument("--retry-max-seconds", type=float, default=300.0)
    parser.add_argument(
        "--min-desired-issues",
        type=int,
        default=1,
        help="Fail if the generated issue plan has fewer desired issues than this",
    )
    parser.add_argument(
        "--planned-action-budget-mode",
        choices=("fail", "warn"),
        default="fail",
        help="Whether planned action budget violations should fail or only warn",
    )
    parser.add_argument(
        "--max-planned-created",
        type=int,
        help="Maximum allowed planned created issues before sync",
    )
    parser.add_argument(
        "--max-planned-updated",
        type=int,
        help="Maximum allowed planned updated issues before sync",
    )
    parser.add_argument(
        "--max-planned-closed",
        type=int,
        help="Maximum allowed planned closed issues before sync",
    )
    parser.add_argument(
        "--max-planned-attached",
        type=int,
        help="Maximum allowed planned sub-issue attachments before sync",
    )
    parser.add_argument(
        "--max-planned-total",
        type=int,
        help="Maximum allowed created+updated+closed+attached issue actions",
    )
    parser.add_argument(
        "--max-planned-stale-parent-closures",
        type=int,
        help="Maximum allowed planned closures for stale managed parent issues",
    )
    parser.add_argument(
        "--max-planned-stale-backlog-closures",
        type=int,
        help="Maximum allowed planned closures for stale managed backlog issues",
    )
    parser.add_argument(
        "--max-planned-stale-extracted-closures",
        type=int,
        help="Maximum allowed planned closures for stale extracted signal issues",
    )
    parser.add_argument(
        "--max-planned-duplicate-marker-closures",
        type=int,
        help="Maximum allowed planned closures for duplicate managed markers",
    )
    return parser.parse_args(argv)


def planned_action_budget_limits_from_args(args: argparse.Namespace) -> dict[str, int]:
    limits = {
        "created": args.max_planned_created,
        "updated": args.max_planned_updated,
        "closed": args.max_planned_closed,
        "attached": args.max_planned_attached,
        "total": args.max_planned_total,
    }
    return {action: limit for action, limit in limits.items() if limit is not None}


def planned_closure_budget_limits_from_args(args: argparse.Namespace) -> dict[str, int]:
    limits = {
        "stale_parent": args.max_planned_stale_parent_closures,
        "stale_backlog": args.max_planned_stale_backlog_closures,
        "stale_extracted": args.max_planned_stale_extracted_closures,
        "duplicate_marker": args.max_planned_duplicate_marker_closures,
    }
    return {category: limit for category, limit in limits.items() if limit is not None}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    matrix_path = args.matrix
    if not matrix_path.is_absolute():
        matrix_path = ROOT / matrix_path
    signals_path = args.signals
    if signals_path is not None and not signals_path.is_absolute():
        signals_path = ROOT / signals_path
    matrix_check_report_path = args.matrix_check_report
    if (
        matrix_check_report_path is not None
        and not matrix_check_report_path.is_absolute()
    ):
        matrix_check_report_path = ROOT / matrix_check_report_path
    matrix_check_report = load_optional_json(matrix_check_report_path)
    manage_sub_issues = not args.no_sub_issues
    planned_action_budget_limits = planned_action_budget_limits_from_args(args)
    planned_closure_budget_limits = planned_closure_budget_limits_from_args(args)

    input_failures = []
    matrix = load_matrix(matrix_path)
    matrix_error = input_load_error(matrix)
    if matrix_error is not None:
        input_failures.append(input_failure_summary("matrix", matrix_path, matrix))
        print("Support issue input is invalid:", file=sys.stderr)
        print(
            "- matrix: {type}: {message}".format(**matrix_error),
            file=sys.stderr,
        )
        if args.plan_output is not None:
            output = (
                args.plan_output
                if args.plan_output.is_absolute()
                else ROOT / args.plan_output
            )
            write_json_report(
                output,
                issue_sync_report(
                    {},
                    mode="dry-run" if args.dry_run else "sync",
                    close_extracted_issues=False,
                    manage_sub_issues=manage_sub_issues,
                    matrix_check_report=matrix_check_report,
                    matrix_check_report_path=matrix_check_report_path,
                    planned_action_budget_limits=planned_action_budget_limits,
                    planned_action_budget_mode=args.planned_action_budget_mode,
                    planned_closure_budget_limits=planned_closure_budget_limits,
                    input_failures=input_failures,
                ),
            )
        return 1

    signals = load_signals(signals_path)
    signals_error = input_load_error(signals)
    if signals_error is not None:
        input_failures.append(input_failure_summary("signals", signals_path, signals))
        print(
            "Support issue signals input is invalid; continuing without signals: "
            "{type}: {message}".format(**signals_error),
            file=sys.stderr,
        )
        signals = None
    desired = build_desired_issues(matrix, signals)
    validation_errors = validate_desired_issues(
        matrix,
        signals,
        desired,
        min_desired_issues=args.min_desired_issues,
    )
    if validation_errors:
        print("Support issue plan is invalid:", file=sys.stderr)
        for message in validation_errors:
            print("- {}".format(message), file=sys.stderr)
        return 1

    token = os.environ.get(args.token_env) or os.environ.get("GH_TOKEN")
    if args.dry_run and not args.inspect_existing:
        token = token or "dry-run-token"
    if not args.repo:
        print("--repo or GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2
    if not token:
        print("{} or GH_TOKEN is required".format(args.token_env), file=sys.stderr)
        return 2

    client = GitHubClient(
        args.repo,
        token,
        api_url=args.api_url,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
    counts = desired_issue_counts(desired)
    print(
        "Desired support issues: total={total}, parents={parents}, backlog={backlog}, extracted={extracted}".format(
            **counts
        )
    )
    close_extracted_issues = signals_allow_extracted_closure(signals)
    if not close_extracted_issues:
        print(
            "Preserving existing extracted support issues because support signals are missing or documentation probes had failures."
        )
    close_pytest_failure_issues = signals_allow_pytest_failure_closure(signals)
    if not close_pytest_failure_issues:
        print(
            "Preserving existing pytest-failure support issues because pytest failure summaries were not provided or could not be loaded."
        )
    existing_issues = None
    existing_sub_issue_ids_by_parent = None
    if args.inspect_existing:
        try:
            existing_issues, existing_sub_issue_ids_by_parent = (
                inspect_existing_issue_state(
                    client,
                    desired,
                    manage_sub_issues=manage_sub_issues,
                )
            )
        except SupportIssueSyncPreflightError as exc:
            print(str(exc), file=sys.stderr)
            if args.plan_output is not None:
                output = (
                    args.plan_output
                    if args.plan_output.is_absolute()
                    else ROOT / args.plan_output
                )
                write_json_report(
                    output,
                    issue_sync_report(
                        desired,
                        mode="dry-run" if args.dry_run else "sync",
                        close_extracted_issues=close_extracted_issues,
                        close_pytest_failure_issues=close_pytest_failure_issues,
                        manage_sub_issues=manage_sub_issues,
                        matrix_check_report=matrix_check_report,
                        matrix_check_report_path=matrix_check_report_path,
                        planned_action_budget_limits=planned_action_budget_limits,
                        planned_action_budget_mode=args.planned_action_budget_mode,
                        planned_closure_budget_limits=planned_closure_budget_limits,
                        input_failures=input_failures,
                        preflight_failure=preflight_failure_summary(exc),
                    ),
                )
            return 1
    planned_action_budget = planned_action_budget_report(
        (
            planned_issue_actions(
                desired,
                existing_issues,
                manage_sub_issues=manage_sub_issues,
                close_extracted_issues=close_extracted_issues,
                close_pytest_failure_issues=close_pytest_failure_issues,
                existing_sub_issue_ids_by_parent=existing_sub_issue_ids_by_parent,
            )
            if existing_issues is not None
            else None
        ),
        planned_action_budget_limits,
        mode=args.planned_action_budget_mode,
    )
    planned_closure_budget = planned_closure_budget_report(
        (
            planned_issue_closures(
                desired,
                existing_issues,
                close_extracted_issues=close_extracted_issues,
                close_pytest_failure_issues=close_pytest_failure_issues,
            )
            if existing_issues is not None
            else None
        ),
        planned_closure_budget_limits,
        mode=args.planned_action_budget_mode,
    )
    if args.plan_output is not None:
        output = (
            args.plan_output
            if args.plan_output.is_absolute()
            else ROOT / args.plan_output
        )
        write_json_report(
            output,
            issue_sync_report(
                desired,
                mode="dry-run" if args.dry_run else "sync",
                close_extracted_issues=close_extracted_issues,
                close_pytest_failure_issues=close_pytest_failure_issues,
                manage_sub_issues=manage_sub_issues,
                matrix_check_report=matrix_check_report,
                matrix_check_report_path=matrix_check_report_path,
                planned_action_budget_limits=planned_action_budget_limits,
                planned_action_budget_mode=args.planned_action_budget_mode,
                planned_closure_budget_limits=planned_closure_budget_limits,
                existing_issues=existing_issues,
                existing_sub_issue_ids_by_parent=existing_sub_issue_ids_by_parent,
                input_failures=input_failures,
            ),
        )
    budget_errors = planned_action_budget_errors(
        planned_action_budget
    ) + planned_closure_budget_errors(planned_closure_budget)
    if budget_errors:
        for message in budget_errors:
            print(message, file=sys.stderr)
        if args.planned_action_budget_mode == "fail":
            return 1
    operation_ledger: list[dict[str, Any]] = []
    try:
        summary = sync_issues(
            client,
            desired,
            dry_run=args.dry_run,
            manage_sub_issues=manage_sub_issues,
            close_extracted_issues=close_extracted_issues,
            close_pytest_failure_issues=close_pytest_failure_issues,
            throttle_seconds=args.throttle_seconds,
            operation_ledger=operation_ledger,
        )
    except SupportIssueSyncMutationError as exc:
        print(str(exc), file=sys.stderr)
        if args.sync_summary_output is not None:
            output = (
                args.sync_summary_output
                if args.sync_summary_output.is_absolute()
                else ROOT / args.sync_summary_output
            )
            write_json_report(
                output,
                issue_sync_report(
                    desired,
                    mode="dry-run" if args.dry_run else "sync",
                    close_extracted_issues=close_extracted_issues,
                    close_pytest_failure_issues=close_pytest_failure_issues,
                    manage_sub_issues=manage_sub_issues,
                    matrix_check_report=matrix_check_report,
                    matrix_check_report_path=matrix_check_report_path,
                    planned_action_budget_limits=planned_action_budget_limits,
                    planned_action_budget_mode=args.planned_action_budget_mode,
                    planned_closure_budget_limits=planned_closure_budget_limits,
                    existing_issues=existing_issues,
                    existing_sub_issue_ids_by_parent=existing_sub_issue_ids_by_parent,
                    sync_summary=exc.summary,
                    sync_failure=sync_failure_summary(exc),
                    input_failures=input_failures,
                    operation_ledger=exc.operation_ledger,
                ),
            )
        return 1
    print(
        "Support issue sync: created={created}, updated={updated}, closed={closed}, attached={attached}, unchanged={unchanged}".format(
            **summary
        )
    )
    if args.sync_summary_output is not None:
        output = (
            args.sync_summary_output
            if args.sync_summary_output.is_absolute()
            else ROOT / args.sync_summary_output
        )
        write_json_report(
            output,
            issue_sync_report(
                desired,
                mode="dry-run" if args.dry_run else "sync",
                close_extracted_issues=close_extracted_issues,
                close_pytest_failure_issues=close_pytest_failure_issues,
                manage_sub_issues=manage_sub_issues,
                matrix_check_report=matrix_check_report,
                matrix_check_report_path=matrix_check_report_path,
                planned_action_budget_limits=planned_action_budget_limits,
                planned_action_budget_mode=args.planned_action_budget_mode,
                planned_closure_budget_limits=planned_closure_budget_limits,
                existing_issues=existing_issues,
                existing_sub_issue_ids_by_parent=existing_sub_issue_ids_by_parent,
                sync_summary=summary,
                input_failures=input_failures,
                operation_ledger=operation_ledger,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
