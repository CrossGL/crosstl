#!/usr/bin/env python3
"""Synchronize PR closing-keyword issue links.

When a pull request title or body uses GitHub closing keywords for same-repo
issues, this tool assigns those issues to the PR author and maintains concise
managed closing lines in the PR body.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

API_VERSION = "2026-03-10"
ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MATRIX_PATH = ROOT / "support" / "generated" / "support-matrix.json"

SECTION_BEGIN = "<!-- crossgl-pr-issue-links:start -->"
SECTION_END = "<!-- crossgl-pr-issue-links:end -->"
SECTION_RE = re.compile(
    rf"\n*{re.escape(SECTION_BEGIN)}\n.*?\n{re.escape(SECTION_END)}\n*",
    re.DOTALL,
)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
KEYWORDS = (
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
)
ISSUE_REF_PATTERN = (
    r"(?:"
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/\d+"
    r"|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+"
    r"|GH-\d+"
    r"|#\d+"
    r")"
)
CLOSING_BLOCK_RE = re.compile(
    r"\b(?:{})\b\s*:?\s+(?P<refs>{}(?:\s*(?:,|and)\s*{})*)".format(
        "|".join(KEYWORDS), ISSUE_REF_PATTERN, ISSUE_REF_PATTERN
    ),
    re.IGNORECASE,
)
REFERENCE_BLOCK_RE = re.compile(
    r"\b(?:refs?|references?)\b\s*:?\s+(?P<refs>{}(?:\s*(?:,|and)\s*{})*)".format(
        ISSUE_REF_PATTERN, ISSUE_REF_PATTERN
    ),
    re.IGNORECASE,
)
ISSUE_REF_RE = re.compile(ISSUE_REF_PATTERN, re.IGNORECASE)
SUPPORT_TRACEABILITY_RE = re.compile(
    r"^\s*Support issue traceability\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
SUPPORT_ISSUE_MARKER_RE = re.compile(
    r"<!--\s*crossgl-support-issue-sync:\s*([^>\s]+)\s*-->"
)
SUPPORT_TRACEABILITY_OPT_OUTS = {
    "coverage only",
    "no issue closed",
    "not closing an issue",
    "not linked",
    "none",
}
SUPPORT_RELEVANT_PATH_PREFIXES = (
    ".github/workflows/",
    "crosstl/backend/",
    "crosstl/translator/codegen/",
    "support/",
    "tests/test_backend/",
    "tests/test_translator/test_codegen/",
    "tests/test_translator/test_frontend_",
    "tools/",
)
SUPPORT_RELEVANT_PATHS = (
    "crosstl/translator/ast.py",
    "crosstl/translator/lexer.py",
    "crosstl/translator/parser.py",
    "crosstl/translator/validation.py",
    "docs/source/support-matrix.rst",
    "examples/test.py",
    "pyproject.toml",
    "requirements.txt",
    "tests/test_translator/test_ast_ir_contracts.py",
    "tests/test_translator/test_backend_contract.py",
    "tests/test_translator/test_ir_legacy_alias_contracts.py",
    "tests/test_translator/test_lexer.py",
    "tests/test_translator/test_parser.py",
    "tests/test_translator/test_translation_pipeline.py",
)
SUPPORT_MATRIX_ARTIFACT_PATH = "support/generated/support-matrix.json"


class GitHubApiError(RuntimeError):
    """Raised when GitHub returns an unexpected API error."""

    def __init__(self, method: str, path: str, status: int, body: str):
        super().__init__(f"{method} {path} failed with {status}: {body}")
        self.method = method
        self.path = path
        self.status = status
        self.body = body


@dataclass(frozen=True)
class PullRequestContext:
    number: int
    title: str
    body: str
    author: str
    changed_files: tuple[str, ...] = ()
    head_repo: str | None = None
    head_sha: str | None = None


@dataclass(frozen=True)
class SupportIssueLink:
    key: str
    issue_numbers: tuple[int, ...]
    reason: str
    base_row: dict[str, Any] | None = None
    head_row: dict[str, Any] | None = None


@dataclass(frozen=True)
class SupportMatrixIssueLinks:
    closure_links: tuple[SupportIssueLink, ...]
    reference_links: tuple[SupportIssueLink, ...]
    missing_closure_keys: tuple[str, ...]
    missing_reference_keys: tuple[str, ...]
    base_backlog_count: int
    head_backlog_count: int
    closure_keys: tuple[str, ...]
    reference_keys: tuple[str, ...]

    @property
    def closure_numbers(self) -> list[int]:
        return flatten_support_issue_numbers(self.closure_links)

    @property
    def reference_numbers(self) -> list[int]:
        return flatten_support_issue_numbers(self.reference_links)


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
        req.add_header("Authorization", f"Bearer {self.token}")
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

    def get_issue(self, number: int) -> dict[str, Any]:
        issue, _ = self.request("GET", f"/repos/{self.repo}/issues/{number}")
        return issue

    def add_issue_assignee(self, number: int, login: str) -> None:
        self.request(
            "POST",
            f"/repos/{self.repo}/issues/{number}/assignees",
            {"assignees": [login]},
        )

    def update_pull_body(self, number: int, body: str) -> None:
        self.request(
            "PATCH",
            f"/repos/{self.repo}/pulls/{number}",
            {"body": body},
        )

    def list_pull_files(self, number: int) -> list[str]:
        files: list[str] = []
        page = 1
        while True:
            payload, _ = self.request(
                "GET",
                f"/repos/{self.repo}/pulls/{number}/files",
                query={"per_page": 100, "page": page},
            )
            batch = payload or []
            files.extend(item["filename"] for item in batch if item.get("filename"))
            if len(batch) < 100:
                return files
            page += 1

    def read_json_file(self, repo: str, path: str, ref: str) -> dict[str, Any]:
        payload, _ = self.request(
            "GET",
            f"/repos/{repo}/contents/{parse.quote(path)}",
            query={"ref": ref},
        )
        if not isinstance(payload, dict):
            raise ValueError("GitHub contents response is not an object")
        encoding = payload.get("encoding")
        content = payload.get("content")
        if encoding != "base64" or not isinstance(content, str):
            raise ValueError("GitHub contents response is not base64 JSON content")
        raw = base64.b64decode(content)
        return json.loads(raw.decode("utf-8"))

    def list_open_support_issues(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        page = 1
        while True:
            payload, _ = self.request(
                "GET",
                f"/repos/{self.repo}/issues",
                query={
                    "labels": "support:matrix",
                    "state": "open",
                    "per_page": 100,
                    "page": page,
                },
            )
            batch = payload or []
            issues.extend(issue for issue in batch if "pull_request" not in issue)
            if len(batch) < 100:
                return issues
            page += 1


def load_pr_context(event_path: Path) -> PullRequestContext:
    event = json.loads(event_path.read_text(encoding="utf-8"))
    pull_request = event.get("pull_request")
    if not pull_request:
        raise ValueError("Event payload does not contain pull_request")
    author = pull_request.get("user", {}).get("login")
    if not author:
        raise ValueError("Pull request payload does not contain user.login")
    return PullRequestContext(
        number=int(pull_request["number"]),
        title=pull_request.get("title") or "",
        body=pull_request.get("body") or "",
        author=author,
        head_repo=pull_request.get("head", {}).get("repo", {}).get("full_name"),
        head_sha=pull_request.get("head", {}).get("sha"),
    )


def strip_managed_section(body: str) -> str:
    return SECTION_RE.sub("\n", body or "").strip()


def strip_unmanaged_managed_link_lines(body: str) -> str:
    lines = []
    for line in (body or "").splitlines():
        candidate = line.strip()
        candidate = re.sub(r"^[-*]\s+", "", candidate)
        if CLOSING_BLOCK_RE.fullmatch(candidate) or REFERENCE_BLOCK_RE.fullmatch(
            candidate
        ):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def strip_code_spans(text: str) -> str:
    text = FENCED_CODE_RE.sub(" ", text)
    return INLINE_CODE_RE.sub(" ", text)


def normalize_issue_ref(ref: str, repo: str) -> int | None:
    owner, repo_name = repo.split("/", 1)
    ref = ref.strip()
    if ref.startswith("#"):
        return int(ref[1:])
    if ref.lower().startswith("gh-"):
        return int(ref[3:])
    url_match = re.fullmatch(
        r"https://github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)",
        ref,
        flags=re.IGNORECASE,
    )
    if url_match:
        ref_owner, ref_repo, number = url_match.groups()
        if ref_owner.lower() == owner.lower() and ref_repo.lower() == repo_name.lower():
            return int(number)
        return None
    repo_match = re.fullmatch(
        r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)#(\d+)",
        ref,
        flags=re.IGNORECASE,
    )
    if repo_match:
        ref_owner, ref_repo, number = repo_match.groups()
        if ref_owner.lower() == owner.lower() and ref_repo.lower() == repo_name.lower():
            return int(number)
    return None


def extract_closing_issue_numbers(title: str, body: str, repo: str) -> list[int]:
    source = "\n".join([title or "", strip_managed_section(body or "")])
    source = strip_code_spans(source)
    numbers: list[int] = []
    seen = set()
    for block in CLOSING_BLOCK_RE.finditer(source):
        refs = block.group("refs")
        for ref_match in ISSUE_REF_RE.finditer(refs):
            number = normalize_issue_ref(ref_match.group(0), repo)
            if number is None or number in seen:
                continue
            seen.add(number)
            numbers.append(number)
    return numbers


def normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def is_support_relevant_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    return normalized in SUPPORT_RELEVANT_PATHS or normalized.startswith(
        SUPPORT_RELEVANT_PATH_PREFIXES
    )


def support_relevant_path_decision(path: str) -> dict[str, str] | None:
    normalized = normalize_repo_path(path)
    if normalized in SUPPORT_RELEVANT_PATHS:
        return {
            "path": normalized,
            "reason": "exact_path",
            "matched": normalized,
        }
    for prefix in SUPPORT_RELEVANT_PATH_PREFIXES:
        if normalized.startswith(prefix):
            return {
                "path": normalized,
                "reason": "prefix",
                "matched": prefix,
            }
    return None


def support_relevant_path_decisions(
    paths: list[str] | tuple[str, ...],
) -> list[dict[str, str]]:
    decisions = []
    for path in paths:
        decision = support_relevant_path_decision(path)
        if decision is not None:
            decisions.append(decision)
    return decisions


def support_matrix_artifact_changed(paths: list[str] | tuple[str, ...]) -> bool:
    return any(
        normalize_repo_path(path) == SUPPORT_MATRIX_ARTIFACT_PATH for path in paths
    )


def has_support_traceability_opt_out(body: str) -> bool:
    match = SUPPORT_TRACEABILITY_RE.search(
        strip_code_spans(strip_managed_section(body))
    )
    if not match:
        return False
    value = match.group("value").strip().lower().rstrip(".")
    return value in SUPPORT_TRACEABILITY_OPT_OUTS


def support_traceability_sources(
    valid_issue_numbers: list[int],
    reference_issue_numbers: list[int],
    body: str,
) -> list[str]:
    sources = []
    if valid_issue_numbers:
        sources.append("same_repo_closing_issue")
    if reference_issue_numbers:
        sources.append("managed_support_issue_reference")
    if has_support_traceability_opt_out(body):
        sources.append("explicit_opt_out")
    return sources


def support_traceability_audit(
    changed_files: list[str] | tuple[str, ...],
    valid_issue_numbers: list[int],
    reference_issue_numbers: list[int],
    body: str,
) -> dict[str, Any]:
    relevant_file_decisions = support_relevant_path_decisions(changed_files)
    sources = support_traceability_sources(
        valid_issue_numbers, reference_issue_numbers, body
    )
    required = bool(relevant_file_decisions)
    satisfied = bool(required and sources)
    failure_reason = None
    if required and not satisfied:
        failure_reason = "missing_issue_or_opt_out"
    elif not required:
        failure_reason = "no_support_relevant_files"
    return {
        "required": required,
        "satisfied": satisfied,
        "satisfaction_sources": sources,
        "failure_reason": failure_reason,
        "support_relevant_files": relevant_file_decisions,
        "support_relevant_count": len(relevant_file_decisions),
        "support_matrix_artifact_changed": support_matrix_artifact_changed(
            changed_files
        ),
        "closing_issue_numbers": list(valid_issue_numbers),
        "managed_reference_issue_numbers": list(reference_issue_numbers),
    }


def managed_section(
    issue_numbers: list[int], reference_issue_numbers: list[int] | None = None
) -> str:
    reference_issue_numbers = reference_issue_numbers or []
    lines = [SECTION_BEGIN]
    lines.extend(f"Closes #{number}" for number in issue_numbers)
    lines.extend(f"Refs #{number}" for number in reference_issue_numbers)
    lines.append(SECTION_END)
    return "\n".join(lines)


def update_body_with_managed_section(
    body: str,
    issue_numbers: list[int],
    reference_issue_numbers: list[int] | None = None,
) -> str:
    reference_issue_numbers = reference_issue_numbers or []
    base = strip_unmanaged_managed_link_lines(strip_managed_section(body or ""))
    if not issue_numbers and not reference_issue_numbers:
        return base
    section = managed_section(issue_numbers, reference_issue_numbers)
    if not base:
        return section
    return base.rstrip() + "\n\n" + section


def support_backlog_keys(matrix: dict[str, Any]) -> set[str]:
    return {
        "backlog:{}:{}".format(item["backend_id"], item["feature_id"])
        for item in matrix.get("backlog", [])
    }


def support_backlog_row_details(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    support_lookup = {}
    for feature in matrix.get("features", []):
        feature_id = feature.get("id")
        for backend_id, support in feature.get("support", {}).items():
            support_lookup[f"backlog:{backend_id}:{feature_id}"] = support or {}

    rows = {}
    for item in matrix.get("backlog", []):
        key = "backlog:{}:{}".format(item["backend_id"], item["feature_id"])
        support = support_lookup.get(key, {})
        details = {}
        for field in ("backend_id", "backend", "feature_id", "feature", "category"):
            if item.get(field):
                details[field] = item[field]
        status = item.get("status") or support.get("status")
        if status:
            details["status"] = status
        notes = item.get("notes") or support.get("notes")
        if notes:
            details["notes"] = notes
        evidence = sorted(support.get("evidence") or [])
        if evidence:
            details["evidence"] = evidence
        rows[key] = details
    return rows


def support_issue_marker_key(issue: dict[str, Any]) -> str | None:
    match = SUPPORT_ISSUE_MARKER_RE.search(issue.get("body") or "")
    return match.group(1) if match else None


def support_issue_numbers_for_keys(
    support_issues: list[dict[str, Any]],
    keys: set[str],
) -> list[int]:
    return flatten_support_issue_numbers(
        support_issue_links_for_keys(
            support_issue_number_lookup(support_issues),
            keys,
            reason="matched_marker_key",
        )
    )


def support_issue_number_lookup(
    support_issues: list[dict[str, Any]],
) -> dict[str, tuple[int, ...]]:
    numbers_by_key: dict[str, set[int]] = {}
    for issue in support_issues:
        marker_key = support_issue_marker_key(issue)
        if marker_key:
            numbers_by_key.setdefault(marker_key, set()).add(int(issue["number"]))
    return {
        key: tuple(sorted(issue_numbers))
        for key, issue_numbers in numbers_by_key.items()
    }


def support_issue_links_for_keys(
    numbers_by_key: dict[str, tuple[int, ...]],
    keys: set[str],
    *,
    reason: str,
    base_rows: dict[str, dict[str, Any]] | None = None,
    head_rows: dict[str, dict[str, Any]] | None = None,
) -> tuple[SupportIssueLink, ...]:
    base_rows = base_rows or {}
    head_rows = head_rows or {}
    return tuple(
        SupportIssueLink(
            key=key,
            issue_numbers=numbers_by_key[key],
            reason=reason,
            base_row=base_rows.get(key),
            head_row=head_rows.get(key),
        )
        for key in sorted(keys)
        if numbers_by_key.get(key)
    )


def support_issue_missing_keys(
    numbers_by_key: dict[str, tuple[int, ...]],
    keys: set[str],
) -> tuple[str, ...]:
    return tuple(key for key in sorted(keys) if not numbers_by_key.get(key))


def flatten_support_issue_numbers(
    links: tuple[SupportIssueLink, ...],
) -> list[int]:
    numbers = [issue_number for link in links for issue_number in link.issue_numbers]
    return sorted(set(numbers))


def support_issue_link_audit(
    links: tuple[SupportIssueLink, ...],
) -> list[dict[str, Any]]:
    entries = []
    for link in links:
        entry = {
            "key": link.key,
            "issues": list(link.issue_numbers),
            "reason": link.reason,
        }
        if link.base_row is not None:
            entry["base_row"] = link.base_row
        if link.head_row is not None:
            entry["head_row"] = link.head_row
        entries.append(entry)
    return entries


def support_matrix_issue_numbers(
    client: GitHubClient,
    pr: PullRequestContext,
    repo: str,
) -> SupportMatrixIssueLinks | None:
    if not pr.head_repo or not pr.head_sha:
        return None
    if not SUPPORT_MATRIX_PATH.exists():
        return None

    base_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    try:
        head_matrix = client.read_json_file(
            pr.head_repo,
            "support/generated/support-matrix.json",
            pr.head_sha,
        )
    except (GitHubApiError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"::warning::Could not inspect PR support matrix links: {exc}")
        return None

    base_backlog_keys = support_backlog_keys(base_matrix)
    head_backlog_keys = support_backlog_keys(head_matrix)
    closure_keys = base_backlog_keys - head_backlog_keys

    base_rows = support_backlog_row_details(base_matrix)
    head_rows = support_backlog_row_details(head_matrix)
    progress_keys = {
        key
        for key in base_backlog_keys & head_backlog_keys
        if base_rows.get(key) != head_rows.get(key)
    }

    if not closure_keys and not progress_keys:
        return SupportMatrixIssueLinks(
            closure_links=(),
            reference_links=(),
            missing_closure_keys=(),
            missing_reference_keys=(),
            base_backlog_count=len(base_backlog_keys),
            head_backlog_count=len(head_backlog_keys),
            closure_keys=(),
            reference_keys=(),
        )

    support_issues = client.list_open_support_issues()
    numbers_by_key = support_issue_number_lookup(support_issues)
    closure_links = support_issue_links_for_keys(
        numbers_by_key,
        closure_keys,
        reason="removed_from_backlog",
        base_rows=base_rows,
        head_rows=head_rows,
    )
    reference_links = support_issue_links_for_keys(
        numbers_by_key,
        progress_keys,
        reason="backlog_row_changed",
        base_rows=base_rows,
        head_rows=head_rows,
    )
    closure_numbers = set(flatten_support_issue_numbers(closure_links))
    reference_links = tuple(
        link
        for link in reference_links
        if not closure_numbers.intersection(link.issue_numbers)
    )
    return SupportMatrixIssueLinks(
        closure_links=closure_links,
        reference_links=reference_links,
        missing_closure_keys=support_issue_missing_keys(numbers_by_key, closure_keys),
        missing_reference_keys=support_issue_missing_keys(
            numbers_by_key, progress_keys
        ),
        base_backlog_count=len(base_backlog_keys),
        head_backlog_count=len(head_backlog_keys),
        closure_keys=tuple(sorted(closure_keys)),
        reference_keys=tuple(sorted(progress_keys)),
    )


def support_closure_issue_numbers(
    client: GitHubClient,
    pr: PullRequestContext,
    repo: str,
) -> list[int]:
    issue_numbers = support_matrix_issue_numbers(client, pr, repo)
    if issue_numbers is None:
        return []
    return issue_numbers.closure_numbers


def empty_support_link_audit(*, inspection_failed: bool) -> dict[str, Any]:
    return {
        "inspection_failed": inspection_failed,
        "closure_links": [],
        "reference_links": [],
        "missing_closure_keys": [],
        "missing_reference_keys": [],
    }


def support_link_audit_from_issue_links(
    issue_links: SupportMatrixIssueLinks,
) -> dict[str, Any]:
    return {
        "inspection_failed": False,
        "matrix_delta": {
            "base_backlog_rows": issue_links.base_backlog_count,
            "head_backlog_rows": issue_links.head_backlog_count,
            "removed_backlog_rows": len(issue_links.closure_keys),
            "changed_backlog_rows": len(issue_links.reference_keys),
            "has_delta": bool(issue_links.closure_keys or issue_links.reference_keys),
        },
        "closure_links": support_issue_link_audit(issue_links.closure_links),
        "reference_links": support_issue_link_audit(issue_links.reference_links),
        "missing_closure_keys": list(issue_links.missing_closure_keys),
        "missing_reference_keys": list(issue_links.missing_reference_keys),
    }


def dedupe_issue_numbers(issue_numbers: list[int]) -> list[int]:
    deduped = []
    seen = set()
    for number in issue_numbers:
        if number in seen:
            continue
        seen.add(number)
        deduped.append(number)
    return deduped


def issue_assignees(issue: dict[str, Any]) -> set[str]:
    return {assignee["login"] for assignee in issue.get("assignees", [])}


def sync_pr_issue_links(
    client: GitHubClient,
    pr: PullRequestContext,
    repo: str,
    *,
    dry_run: bool = False,
    check_support_traceability: bool = False,
    enforce_support_traceability: bool = False,
    sync_support_closures: bool = False,
    sync_support_references: bool = False,
) -> dict[str, Any]:
    issue_numbers = extract_closing_issue_numbers(pr.title, pr.body, repo)
    support_closures: list[int] = []
    support_references: list[int] = []
    support_links_inspected = True
    support_link_audit: dict[str, Any] | None = None
    if sync_support_closures or sync_support_references:
        support_issue_numbers = support_matrix_issue_numbers(client, pr, repo)
        support_links_inspected = support_issue_numbers is not None
        if support_issue_numbers is not None:
            support_link_audit = support_link_audit_from_issue_links(
                support_issue_numbers
            )
            support_closures = support_issue_numbers.closure_numbers
            support_references = support_issue_numbers.reference_numbers
            if not sync_support_closures:
                support_closures = []
            if not sync_support_references:
                support_references = []
        else:
            support_link_audit = empty_support_link_audit(inspection_failed=True)
    issue_numbers = dedupe_issue_numbers(issue_numbers + support_closures)
    reference_issue_numbers = dedupe_issue_numbers(
        [number for number in support_references if number not in set(issue_numbers)]
    )
    summary = {
        "linked": len(issue_numbers),
        "support_closures": len(support_closures),
        "support_references": len(reference_issue_numbers),
        "assigned": 0,
        "assignment_skipped": 0,
        "missing_or_pull": 0,
        "body_updated": 0,
    }
    if support_link_audit is not None:
        summary["support_link_audit"] = support_link_audit
    valid_issue_numbers: list[int] = []

    for number in issue_numbers:
        try:
            issue = client.get_issue(number) if not dry_run else {"number": number}
        except GitHubApiError as exc:
            if exc.status == 404:
                summary["missing_or_pull"] += 1
                continue
            raise
        if "pull_request" in issue:
            summary["missing_or_pull"] += 1
            continue
        valid_issue_numbers.append(number)
        if dry_run or pr.author in issue_assignees(issue):
            continue
        try:
            client.add_issue_assignee(number, pr.author)
            summary["assigned"] += 1
        except GitHubApiError as exc:
            if exc.status in {403, 404, 422}:
                summary["assignment_skipped"] += 1
                continue
            raise

    if support_links_inspected:
        new_body = update_body_with_managed_section(
            pr.body,
            valid_issue_numbers,
            reference_issue_numbers,
        )
        if new_body != (pr.body or ""):
            summary["body_updated"] = 1
            if not dry_run:
                client.update_pull_body(pr.number, new_body)

    if check_support_traceability or enforce_support_traceability:
        changed_files = list(pr.changed_files) or client.list_pull_files(pr.number)
        traceability_audit = support_traceability_audit(
            changed_files,
            valid_issue_numbers,
            reference_issue_numbers,
            pr.body,
        )
        relevant_paths = [
            entry["path"] for entry in traceability_audit["support_relevant_files"]
        ]
        if support_link_audit is not None:
            support_link_audit["changed_files"] = {
                "support_relevant": len(relevant_paths),
                "support_matrix_artifact": int(
                    support_matrix_artifact_changed(changed_files)
                ),
            }
        traceability_required = bool(traceability_audit["required"])
        traceability_satisfied = bool(traceability_audit["satisfaction_sources"])
        summary["traceability_required"] = int(traceability_required)
        summary["traceability_satisfied"] = int(
            traceability_required and traceability_satisfied
        )
        summary["traceability_failed"] = int(
            traceability_required and not traceability_satisfied
        )
        summary["support_relevant_files"] = len(relevant_paths)
        summary["traceability_audit"] = traceability_audit

    return summary


def support_link_audit_counts(audit: dict[str, Any]) -> dict[str, int]:
    closure_links = audit.get("closure_links", [])
    reference_links = audit.get("reference_links", [])
    missing_closure_keys = audit.get("missing_closure_keys", [])
    missing_reference_keys = audit.get("missing_reference_keys", [])
    matrix_delta = audit.get("matrix_delta") or {}
    changed_files = audit.get("changed_files") or {}
    return {
        "closure_candidates": len(closure_links) + len(missing_closure_keys),
        "closure_links": sum(len(entry["issues"]) for entry in closure_links),
        "reference_candidates": len(reference_links) + len(missing_reference_keys),
        "reference_links": sum(len(entry["issues"]) for entry in reference_links),
        "missing_closure_links": len(missing_closure_keys),
        "missing_reference_links": len(missing_reference_keys),
        "inspection_failed": int(bool(audit.get("inspection_failed"))),
        "base_backlog_rows": int(matrix_delta.get("base_backlog_rows", 0)),
        "head_backlog_rows": int(matrix_delta.get("head_backlog_rows", 0)),
        "removed_backlog_rows": int(matrix_delta.get("removed_backlog_rows", 0)),
        "changed_backlog_rows": int(matrix_delta.get("changed_backlog_rows", 0)),
        "matrix_delta_detected": int(bool(matrix_delta.get("has_delta"))),
        "support_relevant_files": int(changed_files.get("support_relevant", 0)),
        "support_matrix_artifact": int(
            bool(changed_files.get("support_matrix_artifact"))
        ),
    }


def format_support_row_delta(entry: dict[str, Any]) -> str:
    if entry.get("reason") == "removed_from_backlog":
        return "row removed"
    pieces = []
    base_row = entry.get("base_row") or {}
    head_row = entry.get("head_row") or {}
    for field in ("status", "notes", "evidence"):
        base_value = base_row.get(field)
        head_value = head_row.get(field)
        if base_value == head_value:
            continue
        if field == "evidence":
            base_value = len(base_value or [])
            head_value = len(head_value or [])
            if base_value == head_value:
                continue
        if field == "notes":
            if base_value and head_value:
                pieces.append("notes changed")
                continue
            base_value = "present" if base_value else "missing"
            head_value = "present" if head_value else "missing"
        pieces.append(
            f"{field}: {base_value or 'missing'} -> {head_value or 'missing'}"
        )
    return "; ".join(pieces)


def format_support_link_entries(entries: list[dict[str, Any]], limit: int = 8) -> str:
    if not entries:
        return "none"
    rendered = []
    for entry in entries[:limit]:
        issues = ", ".join(f"#{number}" for number in entry["issues"])
        reason = entry.get("reason")
        details = [entry["key"]]
        if reason:
            details.append(f"reason={reason}")
        row_delta = format_support_row_delta(entry)
        if row_delta:
            details.append(row_delta)
        rendered.append(f"{issues} ({'; '.join(details)})")
    remaining = len(entries) - limit
    if remaining > 0:
        rendered.append(f"... +{remaining} more")
    return "; ".join(rendered)


def format_support_link_keys(keys: list[str], limit: int = 8) -> str:
    if not keys:
        return "none"
    rendered = list(keys[:limit])
    remaining = len(keys) - limit
    if remaining > 0:
        rendered.append(f"... +{remaining} more")
    return "; ".join(rendered)


def format_issue_numbers(issue_numbers: list[int]) -> str:
    if not issue_numbers:
        return "none"
    return ", ".join(f"#{number}" for number in issue_numbers)


def format_traceability_sources(sources: list[str]) -> str:
    return ", ".join(sources) if sources else "none"


def format_traceability_files(entries: list[dict[str, str]], limit: int = 12) -> str:
    if not entries:
        return "none"
    rendered = [
        "{path} ({reason}:{matched})".format(**entry) for entry in entries[:limit]
    ]
    remaining = len(entries) - limit
    if remaining > 0:
        rendered.append(f"... +{remaining} more")
    return "; ".join(rendered)


def traceability_status(summary: dict[str, Any]) -> str:
    if summary.get("traceability_failed"):
        return "failed"
    if summary.get("traceability_satisfied"):
        return "satisfied"
    if summary.get("traceability_required"):
        return "unsatisfied"
    return "not required"


def emit_support_traceability_audit(summary: dict[str, Any]) -> None:
    audit = summary.get("traceability_audit")
    if not audit:
        return
    print(
        "Support traceability audit: status={status}, sources={sources}, "
        "failure_reason={failure_reason}, closing_issues={closing_issues}, "
        "managed_refs={managed_refs}".format(
            status=traceability_status(summary),
            sources=format_traceability_sources(audit["satisfaction_sources"]),
            failure_reason=audit.get("failure_reason") or "none",
            closing_issues=format_issue_numbers(audit["closing_issue_numbers"]),
            managed_refs=format_issue_numbers(audit["managed_reference_issue_numbers"]),
        )
    )
    print(
        "Support traceability files: "
        + format_traceability_files(audit["support_relevant_files"])
    )


def write_support_traceability_step_summary(summary: dict[str, Any]) -> None:
    audit = summary.get("traceability_audit")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not audit or not step_summary:
        return
    with open(step_summary, "a", encoding="utf-8") as handle:
        handle.write("## Support Traceability\n\n")
        handle.write(f"- Status: {traceability_status(summary)}\n")
        handle.write(
            "- Satisfaction sources: {}\n".format(
                format_traceability_sources(audit["satisfaction_sources"])
            )
        )
        handle.write(
            "- Support-relevant files: {support_relevant_count}\n".format(**audit)
        )
        handle.write(
            "- Support matrix artifact changed: {}\n".format(
                "yes" if audit["support_matrix_artifact_changed"] else "no"
            )
        )
        handle.write(
            "- Closing issue refs: {}\n".format(
                format_issue_numbers(audit["closing_issue_numbers"])
            )
        )
        handle.write(
            "- Managed support refs: {}\n".format(
                format_issue_numbers(audit["managed_reference_issue_numbers"])
            )
        )
        if audit.get("failure_reason"):
            handle.write(f"- Failure reason: {audit['failure_reason']}\n")
        if audit["support_relevant_files"]:
            handle.write("- Files:\n")
            for entry in audit["support_relevant_files"]:
                handle.write("  - `{path}` ({reason}: `{matched}`)\n".format(**entry))
        handle.write("\n")


def emit_support_link_audit(summary: dict[str, Any]) -> None:
    audit = summary.get("support_link_audit")
    if not audit:
        return
    counts = support_link_audit_counts(audit)
    print(
        "Support link audit: closure_candidates={closure_candidates}, "
        "closure_links={closure_links}, reference_candidates={reference_candidates}, "
        "reference_links={reference_links}, missing_closure_links={missing_closure_links}, "
        "missing_reference_links={missing_reference_links}, inspection_failed={inspection_failed}".format(
            **counts
        )
    )
    if audit.get("matrix_delta"):
        print(
            "Support matrix delta: base_backlog_rows={base_backlog_rows}, "
            "head_backlog_rows={head_backlog_rows}, removed_backlog_rows={removed_backlog_rows}, "
            "changed_backlog_rows={changed_backlog_rows}, matrix_delta_detected={matrix_delta_detected}, "
            "support_relevant_files={support_relevant_files}, support_matrix_artifact={support_matrix_artifact}".format(
                **counts
            )
        )
    if audit.get("closure_links"):
        print(
            "Support closure links: "
            + format_support_link_entries(audit["closure_links"])
        )
    if audit.get("reference_links"):
        print(
            "Support reference links: "
            + format_support_link_entries(audit["reference_links"])
        )
    missing_closure_keys = audit.get("missing_closure_keys") or []
    missing_reference_keys = audit.get("missing_reference_keys") or []
    if missing_closure_keys:
        print(
            "::warning::Support backlog rows were removed without open managed "
            "support issues: " + format_support_link_keys(missing_closure_keys)
        )
    if missing_reference_keys:
        print(
            "::warning::Support backlog rows changed without open managed "
            "support issues: " + format_support_link_keys(missing_reference_keys)
        )
    if (
        audit.get("matrix_delta")
        and not counts["inspection_failed"]
        and counts["support_relevant_files"]
        and not counts["matrix_delta_detected"]
    ):
        if counts["support_matrix_artifact"]:
            detail = (
                f"{SUPPORT_MATRIX_ARTIFACT_PATH} changed, but no support backlog "
                "rows were removed or changed."
            )
        else:
            detail = (
                f"{SUPPORT_MATRIX_ARTIFACT_PATH} was not changed, and no support "
                "backlog delta was detected."
            )
        print(
            "::warning::No support matrix backlog delta detected for "
            f"support-relevant PR files: {detail}"
        )


def write_support_link_step_summary(summary: dict[str, Any]) -> None:
    audit = summary.get("support_link_audit")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not audit or not step_summary:
        return
    counts = support_link_audit_counts(audit)
    with open(step_summary, "a", encoding="utf-8") as handle:
        handle.write("## Support Issue Links\n\n")
        handle.write(
            "- Inspection: {}\n".format(
                "failed" if counts["inspection_failed"] else "succeeded"
            )
        )
        handle.write(
            "- Closure links: {closure_links}/{closure_candidates}\n".format(**counts)
        )
        handle.write(
            "- Reference links: {reference_links}/{reference_candidates}\n".format(
                **counts
            )
        )
        if audit.get("matrix_delta"):
            handle.write(
                "- Matrix delta: removed {removed_backlog_rows}, changed {changed_backlog_rows}, "
                "base backlog {base_backlog_rows}, head backlog {head_backlog_rows}\n".format(
                    **counts
                )
            )
        if audit.get("changed_files"):
            handle.write(
                "- Support-relevant files: {support_relevant_files}\n".format(**counts)
            )
            handle.write(
                "- Support matrix artifact changed: {}\n".format(
                    "yes" if counts["support_matrix_artifact"] else "no"
                )
            )
        if audit.get("closure_links"):
            handle.write(
                "- Closes: {}\n".format(
                    format_support_link_entries(audit["closure_links"])
                )
            )
        if audit.get("reference_links"):
            handle.write(
                "- Refs: {}\n".format(
                    format_support_link_entries(audit["reference_links"])
                )
            )
        if audit.get("missing_closure_keys"):
            handle.write(
                "- Missing closure issue links: {}\n".format(
                    format_support_link_keys(audit["missing_closure_keys"])
                )
            )
        if audit.get("missing_reference_keys"):
            handle.write(
                "- Missing reference issue links: {}\n".format(
                    format_support_link_keys(audit["missing_reference_keys"])
                )
            )
        if (
            audit.get("matrix_delta")
            and not counts["inspection_failed"]
            and counts["support_relevant_files"]
            and not counts["matrix_delta_detected"]
        ):
            handle.write(
                "- Attention: no support matrix backlog delta was detected for "
                "support-relevant files; update support/generated/support-matrix.json "
                "or add `Support issue traceability: no issue closed` when intentional.\n"
            )
        handle.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository in OWNER/REPO form",
    )
    parser.add_argument(
        "--event-path",
        type=Path,
        default=(
            Path(os.environ["GITHUB_EVENT_PATH"])
            if os.environ.get("GITHUB_EVENT_PATH")
            else None
        ),
        help="Path to the pull_request event payload",
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
        "--check-support-traceability",
        action="store_true",
        help=(
            "Report whether support-relevant PR files have closing issue refs, "
            "managed support issue refs, or an explicit 'Support issue "
            "traceability: no issue closed' marker"
        ),
    )
    parser.add_argument(
        "--enforce-support-traceability",
        action="store_true",
        help=(
            "Fail when support-relevant PR files lack closing issue refs, managed "
            "support issue refs, or an explicit 'Support issue traceability: no "
            "issue closed' marker"
        ),
    )
    parser.add_argument(
        "--sync-support-closures",
        action="store_true",
        help=(
            "Add closing refs for open managed support issues whose backlog rows "
            "are removed by the PR support matrix"
        ),
    )
    parser.add_argument(
        "--sync-support-references",
        action="store_true",
        help=(
            "Add non-closing refs for open managed support issues whose backlog "
            "rows remain open but changed in the PR support matrix"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    token = os.environ.get(args.token_env) or os.environ.get("GH_TOKEN")
    if args.dry_run:
        token = token or "dry-run-token"
    if not args.repo:
        print("--repo or GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2
    if not args.event_path:
        print("--event-path or GITHUB_EVENT_PATH is required", file=sys.stderr)
        return 2
    if not token:
        print(f"{args.token_env} or GH_TOKEN is required", file=sys.stderr)
        return 2

    pr = load_pr_context(args.event_path)
    client = GitHubClient(args.repo, token, api_url=args.api_url)
    summary = sync_pr_issue_links(
        client,
        pr,
        args.repo,
        dry_run=args.dry_run,
        check_support_traceability=(
            args.check_support_traceability or args.enforce_support_traceability
        ),
        enforce_support_traceability=args.enforce_support_traceability,
        sync_support_closures=args.sync_support_closures,
        sync_support_references=args.sync_support_references,
    )
    print(
        "PR issue link sync: linked={linked}, support_closures={support_closures}, support_references={support_references}, assigned={assigned}, assignment_skipped={assignment_skipped}, missing_or_pull={missing_or_pull}, body_updated={body_updated}".format(
            **summary
        )
    )
    emit_support_link_audit(summary)
    write_support_link_step_summary(summary)
    if args.check_support_traceability or args.enforce_support_traceability:
        print(
            "Support traceability: required={traceability_required}, satisfied={traceability_satisfied}, failed={traceability_failed}, support_relevant_files={support_relevant_files}".format(
                **summary
            )
        )
        emit_support_traceability_audit(summary)
        write_support_traceability_step_summary(summary)
        if summary.get("traceability_failed"):
            warning = (
                "Support-relevant PR changes have no same-repo closing issue "
                "reference, no managed support issue reference, and no explicit "
                "'Support issue traceability: no issue closed' marker."
            )
            print(f"::warning::{warning}")
    if summary.get("traceability_failed"):
        if not args.enforce_support_traceability:
            return 0
        print(
            "Support-relevant PR changes must include a same-repo closing issue "
            "reference, a managed support issue reference, or an explicit "
            "'Support issue traceability: no issue closed' line in the PR body.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
