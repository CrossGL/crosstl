"""What the coverage workflow publishes to the ``coverage`` branch: the report, the chart, ``COVERAGE.md``, a
bounded ``history.csv`` and the trend chart drawn from it.

``python -m tools.coverage publish coverage.json --history old.csv --output-dir site`` writes all five files. Only CI
runs it; contributors never regenerate anything.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from . import chart, ir, trend
from .causes import cause, diagnostic, grouped, headline
from .model import OUTCOMES
from .report import dumps

# One row per language and direction per run; 400 runs is more than a year of nightly runs plus pushes to main.
HISTORY_RUNS = 400
HISTORY_FIELDS = ("date", "commit", "language", "direction", "score", *OUTCOMES)
DETAIL_LIMIT = 160
# A cause in COVERAGE.md is longer, so that two causes do not read the same once cut.
CAUSE_LIMIT = 400
WORKFLOW = "coverage.yml"


def label(row: dict) -> tuple[str, str]:
    if row["language"] == ir.LANGUAGE:
        return chart.name(row["language"]), "expressiveness"
    return chart.name(row["language"]), row["direction"]


def cell(text: str) -> str:
    """``text`` made safe for one Markdown table cell or list item: one line, no pipes, bounded length."""
    line = " ".join(text.split())
    if len(line) > DETAIL_LIMIT:
        line = line[: DETAIL_LIMIT - 1].rstrip() + "…"
    return line.replace("|", "\\|").replace("`", "'")


def _table(report: dict) -> list[str]:
    lines = [
        "| Language | Direction | Score | Supported | Refused | Wrong | Untested |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["backends"]:
        language, direction = label(row)
        if not row["offered"]:
            lines.append(f"| {language} | {direction} | not offered | | | | |")
            continue
        counts = " | ".join(str(row["counts"][o]) for o in OUTCOMES)
        lines.append(f"| {language} | {direction} | {chart.percent(row)} | {counts} |")
    lines += [
        "",
        *(
            f"{chart.name(r['language'])} {r['direction']} is not offered: {r['reason']}."
            for r in report["backends"]
            if not r["offered"]
        ),
    ]
    return lines


def _categories(report: dict) -> list[str]:
    """Supported over measured features per category, one line per offered language and direction.

    Categories with no measured feature in any line are left out.
    """
    rows = [r for r in report["backends"] if r["offered"]]
    categories = sorted(
        {
            f["category"]
            for r in rows
            for f in r["features"]
            if f["outcome"] in ("supported", "refused", "wrong")
        }
    )
    if not categories:
        return []
    lines = [
        "## By category",
        "",
        "Each cell counts the supported features of a category over its supported, refused and wrong ones. Untested "
        "features are left out, and a blank cell means none of the category's features could be measured.",
        "",
        "| Language | Direction | " + " | ".join(categories) + " |",
        "|---|---|" + "---:|" * len(categories),
    ]
    for row in rows:
        language, direction = label(row)
        cells = []
        for category in categories:
            outcomes = Counter(
                f["outcome"] for f in row["features"] if f["category"] == category
            )
            measured = sum(outcomes[o] for o in ("supported", "refused", "wrong"))
            cells.append(f"{outcomes['supported']}/{measured}" if measured else "")
        lines.append(
            "|"
            + "|".join(f" {c} " if c else " " for c in (language, direction, *cells))
            + "|"
        )
    return [*lines, ""]


def _sources(report: dict, repository: str) -> list[str]:
    lines = ["| Language | Feature list | Pinned at |", "|---|---|---|"]
    # A language may draw on several lists, keyed "<language>-<part>"; each row names the language itself.
    for key, source in report["sources"].items():
        language = key.split("-", 1)[0]
        if source.get("repository"):
            url = f"https://github.com/{source['repository']}/blob/{source['commit']}/{source['path']}"
            pinned = f"`{source['commit'][:12]}`"
        else:
            url = f"https://github.com/{repository}/blob/{report['commit'] or 'main'}/{source['path']}"
            pinned = "this repository"
        lines.append(
            f"| {chart.name(language)} | [{cell(source['describes'])}]({url}) | {pinned} |"
        )
    tools = ", ".join(
        f"{tool} {cell(version) if version else '(not installed)'}"
        for tool, version in report["validators"].items()
    )
    return lines + ["", f"Reference tools: {tools}."]


def _executed(report: dict) -> list[str]:
    """Which rows also ran their compute probes on a runtime, as a paragraph, or nothing."""
    rows = [r for r in report["backends"] if r.get("executed")]
    if not rows:
        return []
    ran = "; ".join(
        f"{' '.join(label(r))}, {r['executed']} probes on {r['runtime']}" for r in rows
    )
    return [
        "Where Core's conformance harness has a runtime for the language, a supported compute probe that reads one "
        "input buffer also runs: the native probe and CrossTL's translation get the same inputs, and a "
        f"translation that writes other results is wrong. Executed: {ran}.",
        "",
    ]


def _code(text: str) -> str:
    """``text`` as one bounded Markdown code span, so the ``<probe>`` and ``<feature>`` of a cause stay visible."""
    line = " ".join(text.split())
    if len(line) > CAUSE_LIMIT:
        line = line[: CAUSE_LIMIT - 1].rstrip() + "…"
    return "`" + line.replace("`", "'") + "`"


def _summary(row: dict, text: str, features: list[dict]) -> str:
    """The headline of a cause, or of its one feature's own diagnostic, which names the feature."""
    if len(features) == 1:
        text = diagnostic(features[0])
    if row["language"] == ir.LANGUAGE:
        # An IR gap's diagnostic starts with the language whose probe gave it.
        language, _, rest = text.partition(": ")
        return f"{language}: {headline(rest)}"
    return headline(text)


def _bugs_url(repository: str, row: dict) -> str:
    """The open bug issues of ``row``, which carry the labels :func:`.issues.bug_labels` gives them."""
    labels = (
        "coverage",
        "bug",
        f"backend:{row['language']}",
        f"direction:{row['direction']}",
    )
    query = " ".join(["is:issue", "is:open", *(f"label:{name}" for name in labels)])
    return f"https://github.com/{repository}/issues?q={quote(query, safe=':')}"


def _causes(row: dict, outcome: str) -> list[str]:
    """One list item per cause of ``row``'s ``outcome`` results, the cause shared by most features first."""
    if outcome == "wrong":
        groups = grouped(row)
    else:
        found: dict[str, list[dict]] = {}
        for feature in sorted(row["features"], key=lambda f: f["name"]):
            if feature["outcome"] == outcome:
                found.setdefault(cause(feature, row["language"]), []).append(feature)
        groups = sorted(found.items(), key=lambda item: (-len(item[1]), item[0]))
    lines = []
    for text, features in groups:
        names = ", ".join(f"`{f['name']}`" for f in features)
        count = f" ({len(features)})" if len(features) > 1 else ""
        lines.append(f"- {_code(_summary(row, text, features))}{count}: {names}")
    return lines


def _gaps(row: dict, repository: str) -> list[str]:
    """The refused and wrong features of one row grouped by cause, plus the untested count per category.

    The wrong results are grouped as the bug issues are (:func:`.causes.grouped`), so each item is one bug.
    """
    language, direction = label(row)
    counts = row["counts"]
    untested = Counter(
        f["category"] for f in row["features"] if f["outcome"] == "untested"
    )
    if not counts["refused"] and not counts["wrong"] and not untested:
        return []
    lines = [f"### {language} {direction}", ""]
    if counts["refused"] or counts["wrong"]:
        lines += [
            "<details>",
            f"<summary>{counts['refused'] + counts['wrong']} gaps: {counts['refused']} refused, "
            f"{counts['wrong']} wrong</summary>",
            "",
        ]
        if counts["wrong"]:
            bugs = (
                ""
                if row["language"] == ir.LANGUAGE
                else f" ([open bug issues]({_bugs_url(repository, row)}))"
            )
            lines += [f"Wrong, by cause{bugs}:", "", *_causes(row, "wrong"), ""]
        if counts["refused"]:
            lines += ["Refused, by diagnostic:", "", *_causes(row, "refused"), ""]
        lines += ["</details>", ""]
    if untested:
        tally = ", ".join(f"{category} {n}" for category, n in sorted(untested.items()))
        lines += [f"Untested, by category: {tally}.", ""]
    return lines


def markdown(report: dict, repository: str = "CrossGL/crosstl") -> str:
    """``COVERAGE.md`` for ``report``."""
    commit = report["commit"]
    stamp = (
        f"commit [`{commit[:7]}`](https://github.com/{repository}/commit/{commit})"
        if commit
        else "an uncommitted tree"
    )
    version = f" (CrossTL {report['crosstl']})" if report.get("crosstl") else ""
    lines = [
        "# CrossTL coverage",
        "",
        f"Measured on {stamp} at {report['date']}{version} by the "
        f"[Coverage workflow](https://github.com/{repository}/actions/workflows/{WORKFLOW}), which publishes this "
        "file on every push to `main` and nightly. Run `python -m tools.coverage run` to measure locally.",
        "",
        "Every feature in a pinned upstream feature list gets a generated probe. For output, CrossTL translates a "
        "CrossGL probe to the language; for input, it imports a native probe and translates it back. The "
        "native SPIR-V probe is the module glslang compiles the GLSL probe to, and its round trip must keep the "
        "feature's instruction. The language's reference tool then validates the result:",
        "",
        "- **supported**: the reference tool accepts the translation and, where it runs, the translation "
        "computes what the native probe computes;",
        "- **refused**: CrossTL rejects the feature with a diagnostic;",
        "- **wrong**: CrossTL produces output the reference tool rejects or that computes other results, or fails "
        "without a diagnostic;",
        "- **untested**: no probe can be built for the feature yet, or the reference tool rejects the probe "
        "itself.",
        "",
        "Score = supported / (supported + refused + wrong). Untested features are shown separately and do not "
        "count. The CrossGL IR row covers the union of every language's output features, merged into semantic "
        "features through the SPIR-V each native probe lowers to; a semantic feature is expressible when its "
        "CrossGL probe translates correctly to at least one language that has it.",
        "",
        *_executed(report),
        *_table(report),
        "",
        "## Trend",
        "",
        "![CrossTL coverage over time per language and direction](trend.svg)",
        "",
        *_categories(report),
        "## Feature lists",
        "",
        *_sources(report, repository),
        "",
        "## Gaps",
        "",
        "Each row's wrong and refused features are grouped by diagnostic. A diagnostic drops its line numbers and "
        "writes the probe's own variables as `<probe>`, and one that several features share writes each feature's "
        "name as `<feature>`.",
        "",
    ]
    for row in report["backends"]:
        if row["offered"]:
            lines += _gaps(row, repository)
    return "\n".join(lines).rstrip() + "\n"


def history(report: dict, previous: str = "") -> str:
    """``previous`` history plus this run's rows, keeping the newest :data:`HISTORY_RUNS` runs."""
    rows = [r for r in csv.DictReader(io.StringIO(previous)) if r.get("date")]
    rows = [
        r
        for r in rows
        if (r["date"], r["commit"]) != (report["date"], report["commit"])
    ]
    for row in report["backends"]:
        if not row["offered"]:
            continue
        rows.append(
            {
                "date": report["date"],
                "commit": report["commit"],
                "language": row["language"],
                "direction": row["direction"],
                "score": "" if row["score"] is None else f"{row['score']:.1f}",
                **{o: str(row["counts"][o]) for o in OUTCOMES},
            }
        )
    runs = sorted({(r["date"], r["commit"]) for r in rows})[-HISTORY_RUNS:]
    kept = set(runs)
    out = io.StringIO()
    writer = csv.DictWriter(
        out, HISTORY_FIELDS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(r for r in rows if (r["date"], r["commit"]) in kept)
    return out.getvalue()


def write_site(
    report: dict,
    directory: Path,
    previous_history: str = "",
    repository: str = "CrossGL/crosstl",
) -> list[Path]:
    """Write the five published files into ``directory`` and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    runs = history(report, previous_history)
    files = {
        "coverage.json": dumps(report),
        "coverage.svg": chart.render(report),
        "COVERAGE.md": markdown(report, repository),
        "history.csv": runs,
        "trend.svg": trend.render(runs),
    }
    paths = []
    for filename, text in files.items():
        path = directory / filename
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
