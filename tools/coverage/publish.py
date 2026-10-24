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

from . import chart, ir, trend
from .model import OUTCOMES
from .report import dumps

# One row per language and direction per run; 400 runs is more than a year of nightly runs plus pushes to main.
HISTORY_RUNS = 400
HISTORY_FIELDS = ("date", "commit", "language", "direction", "score", *OUTCOMES)
DETAIL_LIMIT = 160
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


def _gaps(row: dict) -> list[str]:
    """The refused and wrong features of one row, grouped by category, plus the untested count per category."""
    language, direction = label(row)
    gaps = [f for f in row["features"] if f["outcome"] in ("refused", "wrong")]
    untested = Counter(
        f["category"] for f in row["features"] if f["outcome"] == "untested"
    )
    if not gaps and not untested:
        return []
    lines = [f"### {language} {direction}", ""]
    if gaps:
        lines += [
            "<details>",
            f"<summary>{len(gaps)} gaps: {row['counts']['refused']} refused, {row['counts']['wrong']} wrong</summary>",
            "",
        ]
        for category in sorted({f["category"] for f in gaps}):
            lines.append(f"**{category}**")
            lines.append("")
            for feature in sorted(
                (f for f in gaps if f["category"] == category), key=lambda f: f["name"]
            ):
                detail = f": {cell(feature['detail'])}" if feature.get("detail") else ""
                lines.append(f"- `{feature['name']}` {feature['outcome']}{detail}")
            lines.append("")
        lines += ["</details>", ""]
    if untested:
        counts = ", ".join(
            f"{category} {n}" for category, n in sorted(untested.items())
        )
        lines += [f"Untested, by category: {counts}.", ""]
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
    ]
    for row in report["backends"]:
        if row["offered"]:
            lines += _gaps(row)
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
