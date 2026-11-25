"""The GitHub client issue reconciliation (:mod:`.issues`) applies its plans through."""

from __future__ import annotations

import json
import subprocess
import time

from .issues import LABEL, Action, Issue, marker_key


class GitHubError(RuntimeError):
    pass


class GitHub:
    """The few REST calls reconciliation needs, through the ``gh`` command line, which reads ``GH_TOKEN``."""

    # GitHub asks clients to wait at least a second between requests that change something; its secondary rate
    # limit stops a run that closes or comments on many issues at once.
    PAUSE = 1.0

    def __init__(self, repository: str):
        self.repository = repository
        self._changed = -self.PAUSE

    def _api(self, method: str, path: str, payload: dict | None = None):
        if method != "GET":
            time.sleep(max(0.0, self._changed + self.PAUSE - time.monotonic()))
            self._changed = time.monotonic()
        command = ["gh", "api", "--method", method, f"repos/{self.repository}/{path}"]
        return self._gh(command, payload, f"{method} {path}")

    def _query(self, query: str, variables: dict) -> dict:
        """The ``data`` of a GraphQL query, which only reads."""
        payload = {"query": query, "variables": variables}
        found = self._gh(["gh", "api", "graphql"], payload, "graphql")
        return found["data"]

    @staticmethod
    def _gh(command: list[str], payload: dict | None, what: str):
        if payload is not None:
            command = [*command, "--input", "-"]
        done = subprocess.run(
            command,
            input=None if payload is None else json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode:
            raise GitHubError(f"gh api {what}: {done.stderr.strip()}")
        return json.loads(done.stdout) if done.stdout.strip() else None

    # The pull request or commit behind an issue's latest closing; REST issue events leave it out for pull requests.
    _CLOSER = """query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      timelineItems(last: 1, itemTypes: [CLOSED_EVENT]) {
        nodes { ... on ClosedEvent { closer {
          ... on PullRequest { mergeCommit { oid } }
          ... on Commit { oid }
        } } }
      }
    }
  }
}"""

    def closing_commit(self, number: int) -> str:
        """The commit that closed issue ``number`` last: the merge commit of the pull request that fixed it, or the
        commit whose message did; ``""`` when a person or a workflow closed it without one.
        """
        owner, name = self.repository.split("/", 1)
        data = self._query(
            self._CLOSER, {"owner": owner, "name": name, "number": number}
        )
        nodes = data["repository"]["issue"]["timelineItems"]["nodes"]
        closer = (nodes[-1].get("closer") if nodes else None) or {}
        return (closer.get("mergeCommit") or {}).get("oid") or closer.get("oid") or ""

    def contains(self, head: str, commit: str) -> bool:
        """Whether ``commit`` is ``head`` or one of its ancestors."""
        found = self._api("GET", f"compare/{commit}...{head}?per_page=1")
        return found["status"] in ("ahead", "identical")

    def managed(self) -> list[Issue]:
        """Every issue, open or closed, that has the ``coverage`` label and a marker."""
        found: list[Issue] = []
        page = 1
        while True:
            batch = self._api(
                "GET", f"issues?state=all&labels={LABEL}&per_page=100&page={page}"
            )
            for item in batch:
                issue_key = marker_key(item.get("body"))
                if issue_key and "pull_request" not in item:
                    found.append(
                        Issue(
                            item["number"],
                            issue_key,
                            item["state"],
                            item["title"],
                            item.get("body") or "",
                            tuple(label["name"] for label in item["labels"]),
                        )
                    )
            if len(batch) < 100:
                return found
            page += 1

    def apply(self, action: Action) -> int:
        """Carry out ``action`` and return the issue number."""
        if action.kind == "create":
            created = self._api(
                "POST",
                "issues",
                {
                    "title": action.title,
                    "body": action.body,
                    "labels": list(action.labels),
                },
            )
            return created["number"]
        number = action.number
        if action.comment:
            self._api("POST", f"issues/{number}/comments", {"body": action.comment})
        update: dict = {"title": action.title, "body": action.body}
        if action.kind == "close":
            update |= {"state": "closed", "state_reason": action.reason}
        elif action.kind == "reopen":
            update |= {"state": "open"}
        self._api("PATCH", f"issues/{number}", update)
        if action.labels:
            self._api(
                "POST", f"issues/{number}/labels", {"labels": list(action.labels)}
            )
        return number
