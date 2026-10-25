"""How a change moves coverage: the published report against the report of a pull-request run.

``python -m tools.coverage compare published.json coverage.json`` prints Markdown for the pull request's job summary:
the rows whose score changed, then the backend features whose outcome changed, regressions first. A pull request may
skip a leg, such as Metal when it touches no Metal code; rows the new report lacks are named and left out, and the
CrossGL IR row of the published report is rebuilt from the rows both reports share, so a skipped leg does not read as
an IR regression.

The new report may come from a fork, whose run uses the fork's own workflow, and the comment workflow posts the result
with a token that can write comments. So only the fields this module reads are kept, each checked, and the report's
own text (feature names and diagnostics) appears only inside code spans, where no link, mention or HTML renders.

The published coverage lags ``main`` while a Coverage run on ``main`` is still measuring, and a pull request's run
merges it into the ``main`` of its last push. Given the repository, :func:`baseline` finds the ``main`` commit the
pull request was measured on, and the comparison names the pull requests that separate it from the published
coverage, because their changes are in the list too.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from typing import NamedTuple

from . import ir
from .model import DIRECTIONS, LANGUAGES, OUTCOMES
from .publish import DETAIL_LIMIT, label

# Changed features listed in full; a pin bump can change hundreds, and the summary only needs to show the trend.
LIMIT = 40
MISSING = "-"
_KEYS = {(ir.LANGUAGE, ir.DIRECTION)} | {(l, d) for l in LANGUAGES for d in DIRECTIONS}
# Pull requests named when the published coverage and the measured main differ; the rest are counted.
NAMED = 10
_SHA = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[\w.-]+/[\w.-]+")
# A squash merge's subject ends with the number of its pull request.
_PULL = re.compile(r"\(#([0-9]+)\)$")

Api = Callable[[str], dict]


class Baseline(NamedTuple):
    """The commits between the published coverage and the ``main`` a pull request was measured on: those that
    ``main`` has beyond the published commit, and those the published commit has beyond it. Each is spelled as the
    number of its pull request, or as its short hash when its subject names none."""

    ahead: tuple[str, ...] = ()
    behind: tuple[str, ...] = ()


def _gh(path: str) -> dict:
    done = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, timeout=60, check=False
    )
    if done.returncode:
        raise LookupError(done.stderr.strip()[-200:] or f"gh api {path} failed")
    return json.loads(done.stdout)


def _refs(comparison: dict) -> tuple[str, ...]:
    refs = []
    for entry in comparison["commits"]:
        subject = entry["commit"]["message"].partition("\n")[0].strip()
        found = _PULL.search(subject)
        if found:
            refs.append(f"#{found[1]}")
        elif _SHA.fullmatch(entry["sha"]):
            refs.append(f"`{entry['sha'][:12]}`")
        else:
            raise ValueError("the comparison lists a commit without a hash")
    return tuple(refs)


def baseline(
    repository: str, published: str, measured: str, head: str, api: Api | None = None
) -> Baseline | None:
    """The commits between ``published``, the commit of the published coverage, and the ``main`` that a pull
    request's run merged ``head`` into to make ``measured``, the commit it measured.

    ``measured`` comes from the run's report, which a fork can write, so it counts only as a merge of ``head``. None
    when the relation cannot be established.
    """
    if not _REPOSITORY.fullmatch(repository) or not all(
        _SHA.fullmatch(sha) for sha in (published, measured, head)
    ):
        return None
    api = api or _gh
    try:
        merge = api(f"repos/{repository}/commits/{measured}")
        parents = [parent["sha"] for parent in merge["parents"]]
        if len(parents) != 2 or parents[1] != head:
            return None
        if parents[0] == published:
            return Baseline()
        ahead = api(f"repos/{repository}/compare/{published}...{parents[0]}")
        behind = (
            api(f"repos/{repository}/compare/{parents[0]}...{published}")
            if ahead["behind_by"]
            else {"commits": []}
        )
        return Baseline(_refs(ahead), _refs(behind))
    except (
        LookupError,
        TypeError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"warning: no baseline for {measured[:12]}: {error}", file=sys.stderr)
        return None


def _listed(refs: tuple[str, ...]) -> str:
    shown = list(refs[:NAMED])
    if len(refs) > NAMED:
        shown.append(f"{len(refs) - NAMED} more")
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def _commits(refs: tuple[str, ...]) -> str:
    return f"{len(refs)} commit{'s' if len(refs) != 1 else ''}"


def _lag(baseline: Baseline) -> str:
    """Which pull requests the comparison also covers, because the published coverage is not the measured main."""
    sentences = []
    if baseline.ahead:
        sentences.append(
            f"The published coverage is {_commits(baseline.ahead)} behind the `main` this change was measured on, "
            f"so this comparison includes the changes of {_listed(baseline.ahead)}."
        )
    if baseline.behind:
        sentences.append(
            f"This change was measured on a `main` {_commits(baseline.behind)} behind the published coverage, "
            f"so this comparison shows the changes of {_listed(baseline.behind)} reversed."
        )
    return " ".join(sentences)


def _name(key: tuple[str, str]) -> str:
    return " ".join(label({"language": key[0], "direction": key[1]}))


def _code(text: str) -> str:
    """``text`` as one inline code span: one line, bounded, without backticks that could close it."""
    line = " ".join(text.split()).replace("`", "'")
    if len(line) > DETAIL_LIMIT:
        line = line[: DETAIL_LIMIT - 1].rstrip() + "…"
    return f"`{line}`"


def _feature(feature: object) -> dict | None:
    if not isinstance(feature, dict) or not isinstance(feature.get("name"), str):
        return None
    if feature.get("outcome") not in OUTCOMES:
        return None
    detail = feature.get("detail")
    checked = {"name": feature["name"], "outcome": feature["outcome"]}
    return checked | ({"detail": detail} if isinstance(detail, str) else {})


def _row(row: object) -> dict | None:
    if not isinstance(row, dict) or row.get("offered") is not True:
        return None
    if (row.get("language"), row.get("direction")) not in _KEYS:
        return None
    score = row.get("score")
    if score is not None and not (type(score) in (int, float) and 0 <= score <= 100):
        return None
    features = row.get("features")
    features = [_feature(f) for f in features] if isinstance(features, list) else []
    return {
        "language": row["language"],
        "direction": row["direction"],
        "offered": True,
        "score": score,
        "features": [f for f in features if f is not None],
    }


def checked(report: dict) -> dict:
    """The schema and the well-formed rows of known languages in ``report``, reduced to the fields ``compare`` reads."""
    rows = report.get("backends")
    rows = [_row(row) for row in rows] if isinstance(rows, list) else []
    return {
        "schema": report.get("schema"),
        "backends": [row for row in rows if row is not None],
    }


def _rows(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (row["language"], row["direction"]): row
        for row in report["backends"]
        if row.get("offered")
    }


def _comparable(published: dict, current: dict) -> dict[tuple[str, str], dict]:
    """The published rows, restricted to what ``current`` measured, with the IR row rebuilt from those."""
    rows = {
        key: row
        for key, row in _rows(published).items()
        if key in current and key[0] != ir.LANGUAGE
    }
    expressiveness = ir.build(list(rows.values()))
    if expressiveness and (ir.LANGUAGE, ir.DIRECTION) in current:
        rows[ir.LANGUAGE, ir.DIRECTION] = expressiveness
    return rows


def _score(row: dict | None) -> str:
    if row is None:
        return "new"
    return "n/a" if row["score"] is None else f"{row['score']:.1f} %"


def _delta(before: dict | None, after: dict) -> str:
    text = _score(after)
    if before and before["score"] is not None and after["score"] is not None:
        change = after["score"] - before["score"]
        if round(change, 1):
            text += f" ({change:+.1f})"
    return text


def _rank(before: str, after: str) -> int:
    """0 for a regression, 1 for an improvement, 2 for any other change, such as a feature list change or a feature
    no probe could measure."""
    if {before, after} & {MISSING, "untested"}:
        return 2
    if before == "supported" or after == "wrong":
        return 0
    return 1 if after == "supported" else 2


def _changes(
    rows: dict[tuple[str, str], dict], current: dict[tuple[str, str], dict]
) -> list[tuple[int, str]]:
    order = list(current)
    changes = []
    for key, row in current.items():
        # An IR feature changes only when a backend feature does, which is already listed, and a new row is new as a
        # whole.
        if key[0] == ir.LANGUAGE or key not in rows:
            continue
        before = {f["name"]: f for f in rows[key]["features"]}
        after = {f["name"]: f for f in row["features"]}
        for name in sorted(before.keys() | after.keys()):
            old = before.get(name, {}).get("outcome", MISSING)
            new = after.get(name, {}).get("outcome", MISSING)
            if old == new:
                continue
            line = f"{_name(key)} {_code(name)}: {old} → {new}"
            detail = after.get(name, {}).get("detail")
            if detail and new in ("refused", "wrong"):
                line += f" ({_code(detail)})"
            changes.append(((_rank(old, new), order.index(key), name), line))
    return [(key[0], line) for key, line in sorted(changes)]


def compare(
    published: dict, report: dict, limit: int = LIMIT, baseline: Baseline | None = None
) -> str:
    """Markdown describing how ``report`` differs from ``published``, with the pull requests in between when the
    ``baseline`` shows the published coverage was not measured on the same ``main``."""
    head = f"Against the coverage published for `{published['commit'][:12]}` on {published['date'][:10]}"
    report = checked(report)
    if published.get("schema") != report["schema"]:
        return f"{head}: the published report uses another schema, so there is nothing to compare.\n"
    current = _rows(report)
    rows = _comparable(published, current)
    lines = []
    changed = [
        (key, row)
        for key, row in current.items()
        if key not in rows or rows[key]["score"] != row["score"]
    ]
    if changed:
        lines += [
            f"{head}:",
            "",
            "| Backend | Published | This change |",
            "|---|---:|---:|",
        ]
        lines += [
            f"| {_name(key)} | {_score(rows.get(key))} | {_delta(rows.get(key), row)} |"
            for key, row in changed
        ]
    changes = _changes(rows, current)
    if not changed and not changes:
        lines.append(f"{head}, coverage is unchanged.")
    elif not changed:
        lines.append(f"{head}, every score is unchanged.")
    lag = _lag(baseline) if baseline else ""
    if lag:
        lines += ["", lag]
    if changes:
        regressions = sum(1 for rank, _ in changes if rank == 0)
        count = (
            f"{len(changes)} feature{'s' if len(changes) != 1 else ''} change outcome"
        )
        if regressions:
            count += f", {regressions} of them for the worse"
        lines += ["", count + ":", ""]
        lines += [f"- {line}" for _, line in changes[:limit]]
        if len(changes) > limit:
            lines.append(f"- and {len(changes) - limit} more.")
    skipped = [key for key in _rows(published) if key not in current]
    if skipped:
        lines += [
            "",
            "Not measured in this run: " + ", ".join(map(_name, skipped)) + ".",
        ]
    return "\n".join(lines) + "\n"
