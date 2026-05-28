#!/usr/bin/env python3
"""Synchronize PR closing-keyword issue links.

When a pull request title or body uses GitHub closing keywords for same-repo
issues, this tool assigns those issues to the PR author and maintains a small
managed PR body section listing the issues being fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib import error
from urllib import parse
from urllib import request

API_VERSION = "2026-03-10"

SECTION_BEGIN = "<!-- crossgl-pr-issue-links:start -->"
SECTION_END = "<!-- crossgl-pr-issue-links:end -->"
SECTION_RE = re.compile(
    r"\n*{}\n.*?\n{}\n*".format(re.escape(SECTION_BEGIN), re.escape(SECTION_END)),
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
ISSUE_REF_RE = re.compile(ISSUE_REF_PATTERN, re.IGNORECASE)


class GitHubApiError(RuntimeError):
    """Raised when GitHub returns an unexpected API error."""

    def __init__(self, method: str, path: str, status: int, body: str):
        super().__init__("{} {} failed with {}: {}".format(method, path, status, body))
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

    def get_issue(self, number: int) -> dict[str, Any]:
        issue, _ = self.request("GET", "/repos/{}/issues/{}".format(self.repo, number))
        return issue

    def add_issue_assignee(self, number: int, login: str) -> None:
        self.request(
            "POST",
            "/repos/{}/issues/{}/assignees".format(self.repo, number),
            {"assignees": [login]},
        )

    def update_pull_body(self, number: int, body: str) -> None:
        self.request(
            "PATCH",
            "/repos/{}/pulls/{}".format(self.repo, number),
            {"body": body},
        )


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
    )


def strip_managed_section(body: str) -> str:
    return SECTION_RE.sub("\n", body or "").strip()


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


def managed_section(issue_numbers: list[int]) -> str:
    lines = [
        SECTION_BEGIN,
        "## Fixing Issues",
        "",
        "This PR is marked as fixing:",
    ]
    lines.extend("- Fixes #{}".format(number) for number in issue_numbers)
    lines.append(SECTION_END)
    return "\n".join(lines)


def update_body_with_managed_section(body: str, issue_numbers: list[int]) -> str:
    base = strip_managed_section(body or "")
    if not issue_numbers:
        return base
    section = managed_section(issue_numbers)
    if not base:
        return section
    return base.rstrip() + "\n\n" + section


def issue_assignees(issue: dict[str, Any]) -> set[str]:
    return {assignee["login"] for assignee in issue.get("assignees", [])}


def sync_pr_issue_links(
    client: GitHubClient,
    pr: PullRequestContext,
    repo: str,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    issue_numbers = extract_closing_issue_numbers(pr.title, pr.body, repo)
    summary = {
        "linked": len(issue_numbers),
        "assigned": 0,
        "assignment_skipped": 0,
        "missing_or_pull": 0,
        "body_updated": 0,
    }
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

    new_body = update_body_with_managed_section(pr.body, valid_issue_numbers)
    if new_body != (pr.body or ""):
        summary["body_updated"] = 1
        if not dry_run:
            client.update_pull_body(pr.number, new_body)

    return summary


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
        print("{} or GH_TOKEN is required".format(args.token_env), file=sys.stderr)
        return 2

    pr = load_pr_context(args.event_path)
    client = GitHubClient(args.repo, token, api_url=args.api_url)
    summary = sync_pr_issue_links(client, pr, args.repo, dry_run=args.dry_run)
    print(
        "PR issue link sync: linked={linked}, assigned={assigned}, assignment_skipped={assignment_skipped}, missing_or_pull={missing_or_pull}, body_updated={body_updated}".format(
            **summary
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
