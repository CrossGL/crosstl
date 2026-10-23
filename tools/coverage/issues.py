"""Reconcile the coverage tracking and bug issues with a coverage report.

There is one tracking issue per language and direction, plus one for CrossGL IR expressiveness, and one bug issue per
cause of ``wrong`` results in a language and direction (:func:`causes_of`), so one translator bug that breaks many probes
is one issue. Every managed issue carries a hidden marker, ``<!-- crosstl-coverage key=... -->``, and nothing here
edits or closes an issue without one.

:func:`plan` is pure: it compares the report with the managed issues and returns the actions, so a dry run prints
what an applied run would do. :func:`reconcile` applies a plan, then plans once more without filing new bugs, so the
tracking issues link the bugs the run filed and a second run on the same report changes nothing. Bodies carry no
commit or date, so a new measurement edits an issue only when its results change. Issues are closed, never deleted.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace

from . import chart, execution, ir, publish
from .causes import Causes, causes_of, diagnostic, grouped, headline
from .model import ROUND_TRIPS

LABEL = "coverage"
NEW_BUGS = 10
BODY_LIMIT = 60_000
DETAIL_LIMIT = 2_000
MARKER = "<!-- crosstl-coverage key={} -->"
_MARKER = re.compile(r"<!-- crosstl-coverage key=([^\s>]+) -->")
# A bug issue also records the features it covers, so a later run can tell a fix from a result it could not measure.
FEATURES = "<!-- crosstl-coverage features={} -->"
_FEATURES = re.compile(r"<!-- crosstl-coverage features=(.*?) -->")
_UNSAFE = re.compile(r"[^\w.:+-]")
# Bug titles stay short; the body has the whole diagnostic.
TITLE_LIMIT = 120
# A bug issue names this many of the features that share its cause.
LISTED = 40

TOOLS = {
    "cuda": "nvcc",
    "glsl": "glslang",
    "hip": "hipcc",
    "hlsl": "DXC",
    "metal": "the Metal compiler",
    "mojo": "the Mojo compiler",
    "opencl": "clang",
    "rust": "rust-gpu",
    "slang": "slangc",
    "spirv": "spirv-val",
    "webgl": "glslang",
    "wgsl": "naga",
}
AREA = {"output": "area:codegen", "input": "area:frontend"}


@dataclass(frozen=True)
class Issue:
    number: int
    key: str
    state: str
    title: str
    body: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Action:
    """One change to one issue. ``create`` and ``edit`` set the title, body and labels; ``close`` and ``reopen`` do
    the same and change the state, after posting ``comment``."""

    kind: str
    key: str
    title: str
    body: str
    labels: tuple[str, ...] = ()
    number: int | None = None
    comment: str = ""
    reason: str = ""

    def describe(self) -> str:
        target = f"#{self.number}" if self.number else "new issue"
        extra = f" ({self.reason})" if self.reason else ""
        return f"{self.kind:<6} {target:<10} {self.title}{extra}"


@dataclass
class _Plan:
    managed: dict[str, Issue]
    repository: str
    actions: list[Action] = field(default_factory=list)

    def ensure(
        self, key: str, title: str, body: str, labels: tuple[str, ...], comment: str
    ) -> None:
        """Bring the issue for ``key`` to an open state with this title and body, creating it if needed.

        ``comment`` explains a reopening; existing labels are kept and missing ones added.
        """
        issue = self.managed.get(key)
        if issue is None:
            self.actions.append(Action("create", key, title, body, labels))
            return
        missing = tuple(label for label in labels if label not in issue.labels)
        if issue.state == "closed":
            self.actions.append(
                Action("reopen", key, title, body, missing, issue.number, comment)
            )
        elif (issue.title, issue.body) != (title, body) or missing:
            self.actions.append(Action("edit", key, title, body, missing, issue.number))

    def close(
        self, issue: Issue, title: str, body: str, comment: str, reason: str
    ) -> None:
        if issue.state == "open":
            self.actions.append(
                Action(
                    "close", issue.key, title, body, (), issue.number, comment, reason
                )
            )


def key(*parts: str) -> str:
    return "/".join(_UNSAFE.sub("_", part) for part in parts)


def marker_key(body: str | None) -> str:
    """The key in an issue body's marker, or ``""`` when the issue is not managed here."""
    found = _MARKER.search(body or "")
    return found.group(1) if found else ""


def _rows(report: dict) -> list[dict]:
    """Rows with a score; a row that measured nothing, or a language that is not offered, leaves its issues alone."""
    return [
        row
        for row in report["backends"]
        if row.get("offered") and row.get("score") is not None
    ]


def _gaps(row: dict) -> list[dict]:
    return [f for f in row["features"] if f["outcome"] in ("refused", "wrong")]


def _name(row: dict) -> str:
    language, direction = publish.label(row)
    return f"{language} {direction}"


def _destination(language: str) -> str:
    """Where an input probe of ``language`` is translated to, and whose check the translation must pass."""
    emitted = ROUND_TRIPS.get(language)
    if emitted:
        return f"to {chart.name(emitted)} that {TOOLS[emitted]} accepts"
    tool = TOOLS.get(language, language)
    return f"back to code that {tool} accepts"


def _anchor(row: dict) -> str:
    return re.sub(r"[^a-z0-9 -]", "", _name(row).lower()).replace(" ", "-")


def _coverage_url(repository: str, row: dict | None = None) -> str:
    url = f"https://github.com/{repository}/blob/coverage/COVERAGE.md"
    return f"{url}#{_anchor(row)}" if row else url


def tracking_key(row: dict) -> str:
    return key("tracking", row["language"], row["direction"])


def _covered(body: str | None) -> list[str]:
    found = _FEATURES.search(body or "")
    return found.group(1).split(",") if found and found.group(1) else []


def bug_key(row: dict, text: str) -> str:
    """The key of the bug issue for the wrong results of ``row`` whose cause is ``text`` (:func:`causes_of`)."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return key("wrong", row["language"], row["direction"], digest)


def tracking_title(row: dict) -> str:
    gaps = row["counts"]["refused"] + row["counts"]["wrong"]
    noun = "gap" if gaps == 1 else "gaps"
    if row["language"] == ir.LANGUAGE:
        return f"CrossGL IR expressiveness: {chart.percent(row)} ({gaps} {noun})"
    return f"{_name(row)} coverage: {chart.percent(row)} ({gaps} {noun})"


def tracking_labels(row: dict) -> tuple[str, ...]:
    if row["language"] == ir.LANGUAGE:
        return (LABEL, f"backend:{ir.LANGUAGE}", "area:frontend")
    return (
        LABEL,
        f"backend:{row['language']}",
        f"direction:{row['direction']}",
        AREA[row["direction"]],
    )


def bug_labels(row: dict) -> tuple[str, ...]:
    return ("bug", *tracking_labels(row))


def _summary(row: dict, repository: str) -> str:
    counts = row["counts"]
    tally = (
        f"{counts['supported']} supported, {counts['refused']} refused, {counts['wrong']} wrong and "
        f"{counts['untested']} untested"
    )
    link = f"[COVERAGE.md]({_coverage_url(repository, row)})"
    if row["language"] == ir.LANGUAGE:
        return (
            f"CrossGL can express {chart.percent(row)} of the semantic features measured across the "
            f"backends: {tally}. A semantic feature is expressible when the CrossGL probe of at least one backend "
            "feature that lowers to it translates correctly to that backend. A gap here asks for an IR feature "
            f"in `crosstl/translator`. {link} lists every feature."
        )
    tool = TOOLS.get(row["language"], row["language"])
    if row["direction"] == "input":
        rule = (
            f"CrossTL imports its generated {chart.name(row['language'])} probe, which {tool} accepts, "
            f"and translates it {_destination(row['language'])}"
        )
    else:
        rule = f"CrossTL translates its generated CrossGL probe and {tool} accepts the output"
    return (
        f"{_name(row)} scores {chart.percent(row)}: {tally} features. A feature is supported when {rule}. "
        f"{link} lists every feature and the pinned feature list."
    )


def tracking_body(row: dict, repository: str, bugs: dict[str, int]) -> str:
    """The tracking issue body: a summary, then the gaps grouped by category, each with its diagnostic.

    ``bugs`` maps bug keys to existing issue numbers, which are linked from the matching wrong results.
    """
    lines = [
        MARKER.format(tracking_key(row)),
        _summary(row, repository),
        "",
        "The coverage workflow keeps this issue up to date and closes it when every measured feature is supported, "
        "so edits to this description are overwritten. Each wrong result of a backend also has its own `bug` issue.",
    ]
    gaps = sorted(_gaps(row), key=lambda f: (f["category"], f["name"]))
    causes = causes_of(row)
    used = sum(len(line) + 1 for line in lines)
    category = ""
    for index, feature in enumerate(gaps):
        item = (
            [f"### {feature['category']}", ""]
            if feature["category"] != category
            else []
        )
        text = causes.get(feature["name"])
        number = bugs.get(bug_key(row, text)) if text is not None else None
        link = f" (#{number})" if number else ""
        detail = f": {publish.cell(feature['detail'])}" if feature.get("detail") else ""
        item.append(f"- [ ] `{feature['name']}` {feature['outcome']}{link}{detail}")
        cost = sum(len(line) + 1 for line in item) + 2
        if used + cost > BODY_LIMIT:
            lines += ["", f"{len(gaps) - index} more gaps are listed in COVERAGE.md."]
            break
        if item[0].startswith("### "):
            lines.append("")
        lines += item
        used += cost
        category = feature["category"]
    return "\n".join(lines) + "\n"


def _fence(text: str) -> str:
    text = text.strip()
    if len(text) > DETAIL_LIMIT:
        text = text[: DETAIL_LIMIT - 1].rstrip() + "…"
    return text.replace("```", "'''")


def bug_title(row: dict, text: str, features: list[dict]) -> str:
    """The title of the bug for ``features``, whose cause is ``text``; one feature's title keeps its name."""
    first = features[0]
    summary = headline(diagnostic(first) if len(features) == 1 else text)
    if len(summary) > TITLE_LIMIT:
        summary = summary[: TITLE_LIMIT - 1].rstrip() + "…"
    return f"{_name(row)}: {summary}"


def bug_body(row: dict, text: str, features: list[dict], repository: str) -> str:
    """The bug issue body for the ``features`` of ``row`` whose cause is ``text``."""
    language, direction = row["language"], row["direction"]
    tool = TOOLS.get(language, language)
    if direction == "input":
        # A SPIR-V round trip must also keep the feature's instruction (:mod:`.spirv_input`).
        rejects = (
            f"produces code that {tool} rejects or that no longer calls the feature's instruction"
            if language == "spirv"
            else f"produces code that {tool} rejects"
        )
        emitted = ROUND_TRIPS.get(language)
        if emitted:
            rejects = f"produces {chart.name(emitted)} that {TOOLS[emitted]} rejects"
        trip = (
            f"translation to {chart.name(emitted)}"
            if emitted
            else "round trip through CrossGL"
        )
        what = (
            f"CrossTL imports the generated {chart.name(language)} probe, which {tool} accepts, but the "
            f"{trip} fails without a typed diagnostic or {rejects}."
        )
    else:
        what = (
            "CrossTL translates the generated CrossGL probe but fails without a typed diagnostic or "
            f"produces code that {tool} rejects."
        )
    first = features[0]
    executed = execution.found(first.get("detail") or "")
    if executed:
        probe = "CrossGL probe" if direction == "output" else "probe"
        what = (
            f"CrossTL translates the generated {probe} and {tool} accepts the result, but run, it does not "
            f"compute what the native {chart.name(language)} probe computes."
        )
    count = len(features)
    subject = (
        f"The coverage probe for `{first['name']}` ({_name(row)}, {first.get('group') or first['category']}) "
        "gives a wrong result."
        if count == 1
        else f"The coverage probes for {count} {_name(row)} features give a wrong result with the same diagnostic."
    )
    lines = [
        MARKER.format(bug_key(row, text)),
        FEATURES.format(",".join(f["name"] for f in features)),
        f"{subject} {what}",
        "",
        "```text",
        _fence(first.get("detail") or "no diagnostic was recorded"),
        "```",
        "",
    ]
    if count > 1:
        shown = ", ".join(f"`{f['name']}`" for f in features[:LISTED])
        rest = f" and {count - LISTED} more" if count > LISTED else ""
        lines += [f"Affected features: {shown}{rest}.", ""]
    lines += [
        "Reproduce it with:",
        "",
        "```bash",
        f"python -m tools.coverage run --backend {language} --direction {direction} --feature {shlex.quote(first['name'])}"
        + (" --execute" if executed else ""),
        "```",
        "",
        f"The coverage workflow closes this issue when no probe fails this way. [COVERAGE.md]({_coverage_url(repository, row)}) "
        "has the current results.",
    ]
    return "\n".join(lines) + "\n"


def _tracking(plan: _Plan, row: dict, bugs: dict[str, int]) -> None:
    title = tracking_title(row)
    body = tracking_body(row, plan.repository, bugs)
    gaps = row["counts"]["refused"] + row["counts"]["wrong"]
    issue = plan.managed.get(tracking_key(row))
    if gaps:
        plan.ensure(
            tracking_key(row),
            title,
            body,
            tracking_labels(row),
            f"{_name(row)} has {gaps} {'gap' if gaps == 1 else 'gaps'} again, so this issue is reopened.",
        )
    elif issue is not None:
        if issue.state == "open":
            plan.close(
                issue,
                title,
                body,
                "Every measured feature is supported now, so this issue is closed as completed.",
                "completed",
            )
        elif (issue.title, issue.body) != (title, body):
            plan.actions.append(
                Action("edit", issue.key, title, body, (), issue.number)
            )


def _stale(
    plan: _Plan, row: dict, causes: Causes
) -> Iterator[tuple[Issue, list[str], set[str]]]:
    """The row's bug issues whose cause no probe has any more, oldest first.

    Each comes with its features that the row still lists and the keys of the causes those features fail with now.
    An issue with an untested feature is left out: a result that could not be measured says nothing about the bug.
    """
    successor = {f["name"]: bug_key(row, text) for text, group in causes for f in group}
    present = set(successor.values())
    prefix = key("wrong", row["language"], row["direction"]) + "/"
    outcomes = {f["name"]: f["outcome"] for f in row["features"]}
    for issue in sorted(plan.managed.values(), key=lambda i: i.number):
        if not issue.key.startswith(prefix) or issue.key in present:
            continue
        covered = [n for n in _covered(issue.body) if n in outcomes]
        if "untested" not in (outcomes[n] for n in covered):
            yield issue, covered, {successor[n] for n in covered if n in successor}


def _adopt(plan: _Plan, row: dict, causes: Causes) -> None:
    """Give an open bug issue whose features all fail with one other cause, which has no issue yet, that cause's key.

    Its next edit rewrites the marker, so a cause that changes or merges with others keeps its oldest issue.
    """
    for issue, _, failing in list(_stale(plan, row, causes)):
        if issue.state == "open" and len(failing) == 1:
            (new,) = failing
            if new not in plan.managed:
                del plan.managed[issue.key]
                plan.managed[new] = replace(issue, key=new)


def _retire(plan: _Plan, row: dict, causes: Causes) -> None:
    """Close the row's stale bug issues (:func:`_stale`).

    One whose features still fail with other causes is closed in favour of the issues for those causes once they all
    have one.
    """
    tracking = plan.managed.get(tracking_key(row))
    see = f" The remaining gaps are on #{tracking.number}." if tracking else ""
    for issue, covered, failing in list(_stale(plan, row, causes)):
        if failing:
            if not failing <= plan.managed.keys():
                continue
            numbers = sorted(plan.managed[k].number for k in failing)
            links = ", ".join(f"#{number}" for number in numbers)
            comment = f"Its probes still fail, now tracked in {links}, so this issue is closed."
            plan.close(issue, issue.title, issue.body, comment, "not_planned")
        elif covered:
            comment = f"No {_name(row)} probe fails this way any more, so this issue is closed."
            plan.close(issue, issue.title, issue.body, comment + see, "completed")
        else:
            comment = "Its features are no longer in the pinned feature list, so this issue is closed."
            plan.close(issue, issue.title, issue.body, comment, "not_planned")


def _bugs(plan: _Plan, rows: list[tuple[dict, Causes]], new_bugs: int) -> None:
    # New bugs are taken one row at a time, so a capped run files issues for every language rather than the first.
    wanted: list[tuple[int, int, dict, str, list[dict]]] = []
    for order, (row, causes) in enumerate(rows):
        wanted += [
            (rank, order, row, text, group) for rank, (text, group) in enumerate(causes)
        ]
        _retire(plan, row, causes)
    created = 0
    for _, _, row, text, group in sorted(wanted, key=lambda w: (w[0], w[1])):
        issue_key = bug_key(row, text)
        if issue_key not in plan.managed:
            if created >= new_bugs:
                continue
            created += 1
        plan.ensure(
            issue_key,
            bug_title(row, text, group),
            bug_body(row, text, group, plan.repository),
            bug_labels(row),
            "A probe fails this way again, so this issue is reopened.",
        )


def plan(
    report: dict,
    issues: Iterable[Issue],
    repository: str = "CrossGL/crosstl",
    new_bugs: int = NEW_BUGS,
) -> list[Action]:
    """The actions that bring the managed ``issues`` in line with ``report``.

    When two issues carry the same key, the older one is managed and the other is left alone.
    """
    managed: dict[str, Issue] = {}
    for issue in sorted(issues, key=lambda i: i.number):
        if issue.key:
            managed.setdefault(issue.key, issue)
    state = _Plan(managed, repository)
    rows = _rows(report)
    measured = [(row, grouped(row)) for row in rows if row["language"] != ir.LANGUAGE]
    for row, causes in measured:
        _adopt(state, row, causes)
    bugs = {k: i.number for k, i in state.managed.items() if k.startswith("wrong/")}
    for row in rows:
        _tracking(state, row, bugs)
    _bugs(state, measured, new_bugs)
    return state.actions


def reconcile(
    report: dict,
    github: GitHub,
    *,
    new_bugs: int = NEW_BUGS,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> list[Action]:
    """Plan and, unless ``dry_run``, apply, logging each action as it happens; returns the actions."""
    actions = plan(report, github.managed(), github.repository, new_bugs)
    for action in actions:
        if not dry_run:
            github.apply(action)
        log(action.describe())
    if not dry_run and any(action.kind == "create" for action in actions):
        # New bug issues have numbers now; link them from the tracking issues within this run.
        follow_up = plan(report, github.managed(), github.repository, 0)
        for action in follow_up:
            github.apply(action)
            log(action.describe())
        actions += follow_up
    return actions


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
        if payload is not None:
            command += ["--input", "-"]
        done = subprocess.run(
            command,
            input=None if payload is None else json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode:
            raise GitHubError(f"gh api {method} {path}: {done.stderr.strip()}")
        return json.loads(done.stdout) if done.stdout.strip() else None

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
