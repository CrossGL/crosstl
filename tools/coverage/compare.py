"""How a change moves coverage: the published report against the report of a pull-request run.

``python -m tools.coverage compare published.json coverage.json`` prints Markdown for the pull request's job summary:
the rows whose score changed, then the backend features whose outcome changed, regressions first. A pull request may
skip a leg, such as Metal when it touches no Metal code; rows the new report lacks are named and left out, and the
CrossGL IR row of the published report is rebuilt from the rows both reports share, so a skipped leg does not read as
an IR regression.

The new report may come from a fork, whose run uses the fork's own workflow, and the comment workflow posts the result
with a token that can write comments. So only the fields this module reads are kept, each checked, and the report's
own text (feature names and diagnostics) appears only inside code spans, where no link, mention or HTML renders.
"""

from __future__ import annotations

from . import ir
from .model import DIRECTIONS, LANGUAGES, OUTCOMES
from .publish import DETAIL_LIMIT, label

# Changed features listed in full; a pin bump can change hundreds, and the summary only needs to show the trend.
LIMIT = 40
MISSING = "-"
_KEYS = {(ir.LANGUAGE, ir.DIRECTION)} | {(l, d) for l in LANGUAGES for d in DIRECTIONS}


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


def compare(published: dict, report: dict, limit: int = LIMIT) -> str:
    """Markdown describing how ``report`` differs from ``published``."""
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
