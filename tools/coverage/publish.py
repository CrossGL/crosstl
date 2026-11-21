"""What the coverage workflow publishes to the ``coverage`` branch: the report, the chart, ``COVERAGE.md``, a
bounded ``history.csv``, the trend chart drawn from it and one ``pairs/<source>-to-<target>.md`` page with the gaps of
each translation matrix cell (:mod:`.pairs`).

``python -m tools.coverage publish coverage.json --history old.csv --output-dir site`` writes them all, with the
report's feature rows in ``features/`` (see :func:`tools.coverage.report.site`). Only CI runs it; contributors never
regenerate anything.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

from . import chart, ir, trend
from .causes import cause, diagnostic, grouped, headline
from .model import OUTCOMES, ROUND_TRIPS
from .report import site

# One row per language and direction per run; 400 runs is more than a year of nightly runs plus pushes to main.
HISTORY_RUNS = 400
HISTORY_FIELDS = ("date", "commit", "language", "direction", "score", *OUTCOMES)
# The most bytes a published file may have (ADR 0003, section 5).
SIZE_LIMIT = 1_000_000
DETAIL_LIMIT = 160
# A cause in COVERAGE.md is longer, so that two causes do not read the same once cut.
CAUSE_LIMIT = 400
WORKFLOW = "coverage.yml"
# The language ids a matrix page's name may hold, so that a report cannot name a file outside ``pairs/``.
_LANGUAGE_ID = re.compile(r"[a-z0-9]+")


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


def _executed_cell(row: dict) -> str:
    """How many of a row's probes ran and on which runtimes, ``compile-only`` when its language has none, ``not run``
    when the report was measured without running probes, and nothing for the CrossGL IR row, whose outcomes come
    from the output rows."""
    if row["language"] == ir.LANGUAGE:
        return ""
    if row.get("executes") is False:
        return "compile-only"
    if "executed" not in row:
        return "not run"
    if row.get("runtime"):
        return f"{row['executed']} on {row['runtime']}"
    return str(row["executed"])


def _table(report: dict) -> list[str]:
    lines = [
        "| Language | Direction | Score | Supported | Refused | Wrong | Untested | Executed |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["backends"]:
        language, direction = label(row)
        if not row["offered"]:
            lines.append(f"| {language} | {direction} | not offered | | | | | |")
            continue
        counts = " | ".join(str(row["counts"][o]) for o in OUTCOMES)
        lines.append(
            f"| {language} | {direction} | {chart.percent(row)} | {counts} | {_executed_cell(row)} |"
        )
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


def _executed() -> list[str]:
    """What the Executed column of the table counts."""
    return [
        "The Executed column counts the compute probes that also ran on a runtime of Core's conformance harness, "
        "on the GPU or on Mesa's software drivers. A supported compute probe whose translation reads the native "
        "probe's input buffers runs next to the native probe on the same inputs, and a translation that binds "
        "other resources or writes other results is wrong. HLSL, Slang, WGSL and Rust probes run as the SPIR-V "
        "their reference compiler writes for them, and so do GLSL probes that llvmpipe's OpenGL rejects. The "
        "harness has no runtime for the language of a compile-only row, so the reference tool alone decides its "
        "score; 0 means that no supported probe of the row could run.",
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


def _untested(row: dict) -> list[str]:
    """One list item per reason ``row`` gives for leaving features untested, the reason shared by most first.

    A reason is written by the engine or is the reference tool's verdict on the native probe, so it is kept as it
    is: its numbers, such as a shader model, are the reason rather than a position in a probe.
    """
    found: dict[str, list[str]] = {}
    for feature in sorted(row["features"], key=lambda f: f["name"]):
        if feature["outcome"] == "untested":
            reason = (feature.get("detail") or "no reason was recorded").splitlines()[0]
            found.setdefault(reason.strip(), []).append(feature["name"])
    lines = []
    for reason, names in sorted(
        found.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        count = f" ({len(names)})" if len(names) > 1 else ""
        lines.append(f"- {_code(reason)}{count}: " + ", ".join(f"`{n}`" for n in names))
    return lines


def _gaps(row: dict, repository: str, heading: str = "") -> list[str]:
    """The refused and wrong features of one row grouped by cause, and the untested ones grouped by reason.

    The wrong results are grouped as the bug issues are (:func:`.causes.grouped`), so each item is one bug. A
    translation matrix cell is passed as a row of its source language, with its own ``heading``; the coverage
    issues do not file its wrong results as bugs, so it links to none.
    """
    language, direction = label(row)
    counts = row["counts"]
    untested = Counter(
        f["category"] for f in row["features"] if f["outcome"] == "untested"
    )
    if not counts["refused"] and not counts["wrong"] and not untested:
        return []
    lines = [heading or f"### {language} {direction}", ""]
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
                if row["language"] == ir.LANGUAGE or "target" in row
                else f" ([open bug issues]({_bugs_url(repository, row)}))"
            )
            lines += [f"Wrong, by cause{bugs}:", "", *_causes(row, "wrong"), ""]
        if counts["refused"]:
            lines += ["Refused, by diagnostic:", "", *_causes(row, "refused"), ""]
        lines += ["</details>", ""]
    if untested:
        tally = ", ".join(f"{category} {n}" for category, n in sorted(untested.items()))
        lines += [
            "<details>",
            f"<summary>{sum(untested.values())} untested: {tally}</summary>",
            "",
            *_untested(row),
            "",
            "</details>",
            "",
        ]
    return lines


def _stamp(commit: str, date: str, repository: str) -> str:
    where = (
        f"commit [`{commit[:7]}`](https://github.com/{repository}/commit/{commit})"
        if commit
        else "an uncommitted tree"
    )
    return f"{where} at {date}"


def _cell_score(cell: dict) -> str:
    return "n/a" if cell["score"] is None else f"{cell['score']:.1f} %"


def page(cell: dict) -> str:
    """The path, on the coverage branch, of the page with the gaps of the translation matrix cell ``cell``."""
    source, target = cell["source"], cell["target"]
    if not (_LANGUAGE_ID.fullmatch(source) and _LANGUAGE_ID.fullmatch(target)):
        raise ValueError(f"{source!r} to {target!r} is not a cell of the matrix")
    return f"pairs/{source}-to-{target}.md"


def _matrix(report: dict, repository: str) -> list[str]:
    """The ``Translation matrix`` section: one table row per source language and one column per target language."""
    cells = {(c["source"], c["target"]): c for c in report.get("pairs", [])}
    if not cells:
        return []
    offered = [
        r for r in report["backends"] if r["offered"] and r["language"] != ir.LANGUAGE
    ]
    inputs = {r["language"]: r for r in offered if r["direction"] == "input"}
    targets = sorted(
        {r["language"] for r in offered if r["direction"] == "output"}
        | {target for _, target in cells}
    )
    stamp = report.get("matrix") or {"commit": report["commit"], "date": report["date"]}
    lines = [
        "## Translation matrix",
        "",
        "CrossTL translates a file of one language straight to another: the source language's frontend imports it "
        "to CrossGL and the target language's code generator writes it out. A cell measures this for the features "
        "of the source's input row that the target has too, matched by the semantic feature the CrossGL IR row "
        "groups features by. CrossTL translates the source's native probe to the target in one call, and the "
        "target's reference tool checks the result as it checks the target's output row. A feature that the "
        "source's input row leaves untested is untested in the source's cells. Where both languages lower to "
        "SPIR-V (GLSL, HLSL, Rust, Slang, SPIR-V and WGSL), the native probe and its translation also run on the "
        "same runtime and must compute the same results; the other cells are compile-only.",
        "",
        "Rows are source languages and columns are target languages. Each score links to the cell's gaps. A score "
        "in italics is the source's input row, which translates back into the source language (OpenCL C into "
        "HLSL), `none` marks two languages that share no feature, and an empty cell was not measured. The Coverage "
        "workflow measures the matrix nightly; this one was measured on "
        f"{_stamp(stamp.get('commit', ''), stamp.get('date', ''), repository)}.",
        "",
        "| Source \\ Target | " + " | ".join(chart.name(t) for t in targets) + " |",
        "|---|" + "---:|" * len(targets),
    ]
    for source in sorted(inputs):
        entries = []
        for target in targets:
            cell = cells.get((source, target))
            if target == ROUND_TRIPS.get(source, source):
                entries.append(f"*{chart.percent(inputs[source])}*")
            elif cell is None:
                entries.append("")
            elif not cell["features"]:
                entries.append("none")
            else:
                entries.append(f"[{_cell_score(cell)}]({page(cell)})")
        lines.append(f"| {chart.name(source)} | " + " | ".join(entries) + " |")
    return [*lines, ""]


def cell_markdown(cell: dict, report: dict, repository: str = "CrossGL/crosstl") -> str:
    """The page :func:`page` names: the score and counts of ``cell`` and its gaps, grouped as a row's are."""
    source, target = chart.name(cell["source"]), chart.name(cell["target"])
    stamp = report.get("matrix") or {"commit": report["commit"], "date": report["date"]}
    counts = cell["counts"]
    executes = (
        "Its compute probes also ran on a runtime"
        if cell.get("executes")
        else "It is compile-only"
    )
    lines = [
        f"# CrossTL translation matrix: {source} to {target}",
        "",
        f"Measured on {_stamp(stamp.get('commit', ''), stamp.get('date', ''), repository)} by the "
        f"[Coverage workflow](https://github.com/{repository}/actions/workflows/{WORKFLOW}). CrossTL translated "
        f"each native {source} probe whose feature {target} has too straight to {target}, and "
        f"{cell.get('validator') or 'the reference tool'} checked the result. {executes}. "
        "[COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.",
        "",
        f"Score {_cell_score(cell)}: "
        + ", ".join(f"{counts[o]} {o}" for o in OUTCOMES)
        + ".",
        "",
    ]
    row = {**cell, "language": cell["source"], "direction": "pair"}
    gaps = _gaps(row, repository, heading="## Gaps")
    return "\n".join([*lines, *(gaps or ["No gaps."])]).rstrip() + "\n"


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
        "- **supported**: the reference tool accepts the translation, which keeps the built-ins and decorations "
        "the feature needs once compiled and, where it runs, computes what the native probe computes;",
        "- **refused**: CrossTL rejects the feature with a diagnostic;",
        "- **wrong**: CrossTL produces output the reference tool rejects, that loses a built-in or decoration the "
        "feature needs or that computes other results, writes a placeholder comment in place of the feature, or "
        "fails without a diagnostic;",
        "- **untested**: no probe can be built for the feature yet, or the reference tool rejects the probe "
        "itself.",
        "",
        "Score = supported / (supported + refused + wrong). Untested features are shown separately and do not "
        "count. The CrossGL IR row covers the union of every language's output features, merged into semantic "
        "features through the SPIR-V each native probe lowers to; a semantic feature is expressible when its "
        "CrossGL probe translates correctly to at least one language that has it, or to valid SPIR-V that lowers "
        "to the same semantic feature.",
        "",
        *_executed(),
        *_table(report),
        "",
        *_matrix(report, repository),
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
        "name as `<feature>`. Untested features are grouped by the reason no probe could measure them.",
        "",
    ]
    for row in report["backends"]:
        if row["offered"]:
            lines += _gaps(row, repository)
    return "\n".join(lines).rstrip() + "\n"


def _csv(rows: Iterable[dict], header: bool = True) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(
        out, HISTORY_FIELDS, extrasaction="ignore", lineterminator="\n"
    )
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def history(report: dict, previous: str = "") -> str:
    """``previous`` history plus this run's rows, keeping the newest :data:`HISTORY_RUNS` runs that fit in
    :data:`SIZE_LIMIT` bytes, so that a growing number of rows shortens the history instead of stopping publication.
    """
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
    sizes: dict[tuple[str, str], int] = {}
    for r in rows:
        run = (r["date"], r["commit"])
        sizes[run] = sizes.get(run, 0) + len(_csv([r], header=False).encode("utf-8"))
    size = len(_csv([]).encode("utf-8"))
    kept = set()
    for run in sorted(sizes, reverse=True)[:HISTORY_RUNS]:
        size += sizes[run]
        if size > SIZE_LIMIT:
            break
        kept.add(run)
    return _csv(r for r in rows if (r["date"], r["commit"]) in kept)


def write_site(
    report: dict,
    directory: Path,
    previous_history: str = "",
    repository: str = "CrossGL/crosstl",
) -> list[Path]:
    """Write the published files into ``directory`` and return their paths.

    Raises ``ValueError`` before writing anything if a file would exceed :data:`SIZE_LIMIT`.
    """
    runs = history(report, previous_history)
    files = {
        **site(report),
        "coverage.svg": chart.render(report),
        "COVERAGE.md": markdown(report, repository),
        "history.csv": runs,
        "trend.svg": trend.render(runs),
        **{
            page(cell): cell_markdown(cell, report, repository)
            for cell in report.get("pairs", [])
        },
    }
    for filename, text in files.items():
        size = len(text.encode("utf-8"))
        if size > SIZE_LIMIT:
            raise ValueError(
                f"{filename} would be {size} bytes, over the {SIZE_LIMIT}-byte limit for a published file"
            )
    paths = []
    for filename, text in files.items():
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
