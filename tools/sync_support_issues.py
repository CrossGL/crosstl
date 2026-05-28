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

    def __init__(self, method: str, path: str, status: int, body: str):
        super().__init__("{} {} failed with {}: {}".format(method, path, status, body))
        self.method = method
        self.path = path
        self.status = status
        self.body = body


@dataclass(frozen=True)
class DesiredIssue:
    key: str
    title: str
    body: str
    labels: tuple[str, ...]
    parent_key: str | None = None


def load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_signals(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
            hit.get("symbol")
            or hit.get("path")
            or hit.get("source")
            or hit.get("term")
            or "hit"
        )
        terms = ", ".join(hit.get("matched_terms", [])) or hit.get("term") or ""
        count = hit.get("count")
        details = terms or "matched"
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


class GitHubClient:
    def __init__(self, repo: str, token: str, api_url: str = "https://api.github.com"):
        if "/" not in repo:
            raise ValueError("Repository must be in OWNER/REPO form")
        self.repo = repo
        self.api_url = api_url.rstrip("/")
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, str]]:
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

        try:
            with request.urlopen(req, timeout=30) as response:
                text = response.read().decode("utf-8")
                data = json.loads(text) if text else None
                return data, {
                    key.lower(): value for key, value in response.headers.items()
                }
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise GitHubApiError(method, path, exc.code, error_body) from exc

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


def issue_labels(issue: dict[str, Any]) -> set[str]:
    return {label["name"] for label in issue.get("labels", [])}


def sync_issues(
    client: GitHubClient,
    desired: dict[str, DesiredIssue],
    *,
    dry_run: bool = False,
    manage_sub_issues: bool = True,
    throttle_seconds: float = 0.2,
) -> dict[str, int]:
    summary = {
        "created": 0,
        "updated": 0,
        "closed": 0,
        "attached": 0,
        "unchanged": 0,
    }
    existing_issues = client.list_managed_issues() if not dry_run else []
    existing_by_key = {}
    for issue in existing_issues:
        key = marker_key(issue.get("body"))
        if key:
            existing_by_key[key] = issue

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
        client.ensure_label(label, color, description)
        time.sleep(throttle_seconds)

    materialized: dict[str, dict[str, Any]] = {}
    for key, target in desired.items():
        issue = existing_by_key.get(key)
        if issue is None:
            issue = client.create_issue(target)
            summary["created"] += 1
            time.sleep(throttle_seconds)
        else:
            before = (
                issue.get("title"),
                issue.get("body"),
                issue.get("state"),
                issue_labels(issue),
            )
            issue = client.update_issue(issue, target)
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
                child_ids_by_parent[parent_number] = {
                    item["id"] for item in client.list_sub_issues(parent)
                }
                time.sleep(throttle_seconds)
            if child["id"] in child_ids_by_parent[parent_number]:
                continue
            client.add_sub_issue(parent, child)
            child_ids_by_parent[parent_number].add(child["id"])
            summary["attached"] += 1
            time.sleep(throttle_seconds)

    for key, issue in existing_by_key.items():
        if key in desired:
            continue
        if not key.startswith(("backlog:", "parent:", "extracted:")):
            continue
        if issue.get("state") == "closed":
            summary["unchanged"] += 1
            continue
        client.close_issue(issue, closed_body(issue, key))
        summary["closed"] += 1
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
    parser.add_argument("--no-sub-issues", action="store_true")
    parser.add_argument("--throttle-seconds", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    matrix_path = args.matrix
    if not matrix_path.is_absolute():
        matrix_path = ROOT / matrix_path
    matrix = load_matrix(matrix_path)
    signals_path = args.signals
    if signals_path is not None and not signals_path.is_absolute():
        signals_path = ROOT / signals_path
    desired = build_desired_issues(matrix, load_signals(signals_path))

    token = os.environ.get(args.token_env) or os.environ.get("GH_TOKEN")
    if args.dry_run:
        token = token or "dry-run-token"
    if not args.repo:
        print("--repo or GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2
    if not token:
        print("{} or GH_TOKEN is required".format(args.token_env), file=sys.stderr)
        return 2

    client = GitHubClient(args.repo, token, api_url=args.api_url)
    summary = sync_issues(
        client,
        desired,
        dry_run=args.dry_run,
        manage_sub_issues=not args.no_sub_issues,
        throttle_seconds=args.throttle_seconds,
    )
    print(
        "Support issue sync: created={created}, updated={updated}, closed={closed}, attached={attached}, unchanged={unchanged}".format(
            **summary
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
