"""One documentation page per language, written from the published coverage report when the documentation is built.

The Sphinx extension ``docs/source/_ext/coverage_pages.py`` calls :func:`write` from its ``builder-inited`` hook. It
reads the report from a directory that holds the files of the ``coverage`` branch, or downloads ``coverage.json`` and
the feature files it names from :data:`PUBLISHED` with :func:`fetch`. The pages are Markdown, which the documentation
reads with MyST, and go to ``docs/source/languages/``, which git ignores, so nobody commits or regenerates them.

Each page has the language's rows of the report: the score and counts of each direction, the reference tool and the
runtime the probes ran on, the pinned feature lists, the measured features per category and the gaps grouped by cause,
as ``COVERAGE.md`` groups them.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from . import chart, ir, report
from .publish import (
    WORKFLOW,
    _bugs_url,
    _causes,
    _executed_cell,
    _untested,
    cell,
    label,
)

# Where the Coverage workflow publishes the report and its feature files.
PUBLISHED = "https://raw.githubusercontent.com/CrossGL/crosstl/coverage/"
COVERAGE_MD = "https://github.com/CrossGL/crosstl/blob/coverage/COVERAGE.md"
# The only names a page may have, so that none leads outside its directory. A downloaded file must have a name
# that report.load would read: ``report.FEATURE_FILE`` for a row, ``report.CELL_FILE`` for a matrix cell.
_LANGUAGE_ID = re.compile(r"[a-z0-9]+")
# The most bytes one downloaded file may have; the Coverage workflow publishes none over 1 MB.
_DOWNLOAD_LIMIT = 4_000_000
HEADER = "% Written from coverage.json by tools/coverage/docs.py when the documentation is built. Do not edit."
DIRECTIONS = ("output", "input")


def _download(url: str, opener: Callable) -> bytes:
    with opener(url, timeout=60) as response:
        data = response.read(_DOWNLOAD_LIMIT + 1)
    if len(data) > _DOWNLOAD_LIMIT:
        raise ValueError(f"{url} is over {_DOWNLOAD_LIMIT} bytes")
    return data


def fetch(
    base: str,
    directory: Path,
    opener: Callable = urllib.request.urlopen,
) -> Path:
    """Download ``coverage.json`` from ``base``, a URL ending in ``/``, and every file it names into ``directory``.

    Returns the path of the downloaded ``coverage.json``. Raises ``ValueError`` if a row names a file other than a
    ``features/<language>-<direction>.json`` or a matrix cell one other than a
    ``features/pairs/<source>-to-<target>.json``.
    """
    text = _download(base + "coverage.json", opener)
    data = json.loads(text)
    names = []
    entries = [
        *((report.FEATURE_FILE, entry) for entry in data.get("backends", [])),
        *((report.CELL_FILE, entry) for entry in data.get("pairs", [])),
    ]
    for pattern, entry in entries:
        name = entry.get("file") if isinstance(entry, dict) else None
        if name is None:
            continue
        if not isinstance(name, str) or not pattern.fullmatch(name):
            raise ValueError(
                f"coverage.json names {name!r}, which is not a published file"
            )
        names.append(name)
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_download(base + name, opener))
    path = directory / "coverage.json"
    path.write_bytes(text)
    return path


def load(source: str, opener: Callable = urllib.request.urlopen) -> dict:
    """The report at ``source``: a directory with the files of the ``coverage`` branch, or the URL they are under."""
    if not source.startswith(("https://", "http://")):
        return report.load(Path(source) / "coverage.json")
    directory = Path(tempfile.mkdtemp(prefix="crosstl-coverage-"))
    try:
        return report.load(fetch(source.rstrip("/") + "/", directory, opener))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def languages(data: dict) -> list[str]:
    """The languages of the report, in the order its rows first name them."""
    found = []
    for row in data["backends"]:
        if row["language"] not in found:
            if not _LANGUAGE_ID.fullmatch(row["language"]):
                raise ValueError(f"{row['language']!r} is not the name of a language")
            found.append(row["language"])
    return found


def _rows(data: dict, language: str) -> list[dict]:
    rows = [r for r in data["backends"] if r["language"] == language]
    order = {d: i for i, d in enumerate(DIRECTIONS)}
    return sorted(rows, key=lambda r: order.get(r["direction"], len(order)))


def _stamp(data: dict, repository: str) -> str:
    commit = data.get("commit") or ""
    where = (
        f"commit [`{commit[:7]}`](https://github.com/{repository}/commit/{commit})"
        if commit
        else "an uncommitted tree"
    )
    version = f" (CrossTL {data['crosstl']})" if data.get("crosstl") else ""
    return f"{where} at {data.get('date', '')}{version}"


def _tool(data: dict, row: dict) -> str:
    tool = row.get("validator")
    if not tool:
        return ""
    version = data.get("validators", {}).get(tool)
    if not version:
        return cell(f"{tool} (not installed)")
    # Most tools print their own name first ("naga 30.0.1"); glslang and DXC print another one.
    return cell(
        version if version.lower().startswith(tool.lower()) else f"{tool} ({version})"
    )


def _table(data: dict, rows: list[dict]) -> list[str]:
    lines = [
        "| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        _, direction = label(row)
        if not row["offered"]:
            lines.append(f"| {direction} | not offered | | | | | | |")
            continue
        counts = row["counts"]
        lines.append(
            f"| {direction} | {chart.percent(row)} | {counts['supported']} | {counts['refused']} | "
            f"{counts['wrong']} | {counts['untested']} | {_executed_cell(row)} | {_tool(data, row)} |"
        )
    reasons = [
        f"{chart.name(r['language'])} {r['direction']} is not offered: {r['reason']}."
        for r in rows
        if not r["offered"] and r.get("reason")
    ]
    return [*lines, "", *reasons, *([""] if reasons else [])]


def _sources(data: dict, language: str, repository: str) -> list[str]:
    """The pinned feature lists of ``language``, keyed ``<language>`` or ``<language>-<part>`` in the report."""
    lines = []
    for key, source in data.get("sources", {}).items():
        if key.split("-", 1)[0] != language:
            continue
        if source.get("repository"):
            url = f"https://github.com/{source['repository']}/blob/{source['commit']}/{source['path']}"
            pinned = f"pinned at `{source['commit'][:12]}`"
        else:
            url = f"https://github.com/{repository}/blob/{data.get('commit') or 'main'}/{source['path']}"
            pinned = "kept in this repository"
        lines.append(f"- [{cell(source['describes'])}]({url}), {pinned}")
    if not lines:
        return []
    return ["## Feature lists", "", *lines, ""]


def _categories(rows: list[dict]) -> list[str]:
    """Supported over measured features per category, one column per offered direction."""
    rows = [r for r in rows if r["offered"]]
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
        "Each cell counts the supported features of a category over its supported, refused and wrong ones.",
        "",
        "| Category | " + " | ".join(label(r)[1] for r in rows) + " |",
        "|---|" + "---:|" * len(rows),
    ]
    for category in categories:
        cells = []
        for row in rows:
            outcomes = Counter(
                f["outcome"] for f in row["features"] if f["category"] == category
            )
            measured = sum(outcomes[o] for o in ("supported", "refused", "wrong"))
            cells.append(f"{outcomes['supported']}/{measured}" if measured else "")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _gaps(row: dict, repository: str) -> list[str]:
    """The wrong and refused features of ``row`` grouped by cause, and the untested ones grouped by reason."""
    counts = row["counts"]
    if not counts["refused"] and not counts["wrong"] and not counts["untested"]:
        return []
    _, direction = label(row)
    lines = [f"### {direction.capitalize()}", ""]
    if counts["wrong"]:
        bugs = (
            ""
            if row["language"] == ir.LANGUAGE
            else f" ([open bug issues]({_bugs_url(repository, row)}))"
        )
        lines += [f"Wrong, by cause{bugs}:", "", *_causes(row, "wrong"), ""]
    if counts["refused"]:
        lines += ["Refused, by diagnostic:", "", *_causes(row, "refused"), ""]
    if counts["untested"]:
        lines += ["Untested, by reason:", "", *_untested(row), ""]
    return lines


def page(
    data: dict,
    language: str,
    guide: str | None = None,
    repository: str = "CrossGL/crosstl",
) -> str:
    """The Markdown page of ``language``; ``guide`` is the document name of its backend guide, if it has one."""
    rows = _rows(data, language)
    if not rows:
        raise ValueError(f"the report has no row for {language!r}")
    name = chart.name(language)
    offered = [r for r in rows if r["offered"]]
    lines = [
        HEADER,
        "",
        f"# {name}",
        "",
        f"Measured on {_stamp(data, repository)} by the "
        f"[Coverage workflow](https://github.com/{repository}/actions/workflows/{WORKFLOW}). The documentation "
        f"build writes this page from the published `coverage.json`; [COVERAGE.md]({COVERAGE_MD}) defines each "
        "outcome. Score = supported / (supported + refused + wrong).",
        "",
    ]
    if language == ir.LANGUAGE:
        lines += [
            "The CrossGL IR row counts the semantic features of every measured output language that CrossGL can "
            "express: a feature is expressible when its CrossGL probe translates correctly to at least one language "
            "that has it.",
            "",
        ]
    if guide:
        lines += [
            f"The {{doc}}`backend guide </{guide}>` describes how CrossTL translates {name}.",
            "",
        ]
    lines += _table(data, rows)
    lines += _sources(data, language, repository)
    lines += _categories(rows)
    gaps = [line for row in offered for line in _gaps(row, repository)]
    if gaps:
        lines += [
            "## Gaps",
            "",
            "A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that "
            "several features share writes each feature's name as `<feature>`.",
            "",
            *gaps,
        ]
    return "\n".join(lines).rstrip() + "\n"


def _score(rows: list[dict], direction: str) -> str:
    for row in rows:
        if label(row)[1] == direction:
            return chart.percent(row)
    return ""


def index(data: dict, repository: str = "CrossGL/crosstl") -> str:
    """The page that lists every language with its scores and links to the language's page."""
    lines = [
        HEADER,
        "",
        "# Coverage by language",
        "",
        "Each page lists what CrossTL translates to and from one language, measured on "
        f"{_stamp(data, repository)}. CI probes every feature of a pinned upstream feature list, checks each "
        "translation with the language's reference compiler or validator and, where it has a runtime for the "
        f"language, runs it. [COVERAGE.md]({COVERAGE_MD}) has the same numbers for every language on one page.",
        "",
        "| Language | Output | Input |",
        "|---|---:|---:|",
    ]
    names = languages(data)
    for language in names:
        rows = _rows(data, language)
        output = _score(rows, "expressiveness" if language == ir.LANGUAGE else "output")
        lines.append(
            f"| [{chart.name(language)}]({language}.md) | {output} | {_score(rows, 'input')} |"
        )
    lines += [
        "",
        "```{toctree}",
        ":hidden:",
        "",
        *names,
        "```",
    ]
    return "\n".join(lines) + "\n"


def unavailable(source: str, error: str) -> str:
    """The index page of a build that could not read the report, which links to the published one instead."""
    return "\n".join(
        [
            HEADER,
            "",
            "# Coverage by language",
            "",
            f"This build could not read the coverage report from `{cell(source)}` ({cell(error)}), so it has no "
            f"page per language. [COVERAGE.md]({COVERAGE_MD}) has the published numbers. Set `CROSSTL_COVERAGE` to "
            "a directory with the files of the `coverage` branch to build the pages offline.",
            "",
        ]
    )


def pages(
    data: dict, guides: dict[str, str], repository: str = "CrossGL/crosstl"
) -> dict[str, str]:
    """Every page by file name: ``index.md`` and one ``<language>.md`` per language of ``data``."""
    files = {"index.md": index(data, repository)}
    for language in languages(data):
        files[f"{language}.md"] = page(data, language, guides.get(language), repository)
    return files


def write(directory: Path, files: dict[str, str]) -> list[Path]:
    """Write ``files`` into ``directory``, leaving unchanged files alone and removing the pages no longer listed.

    Returns the paths it wrote.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in files.items():
        if "/" in name or not name.endswith(".md"):
            raise ValueError(f"{name!r} is not the name of a page")
        path = directory / name
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
            written.append(path)
    for stale in _stale(directory, files):
        stale.unlink()
    return written


def _stale(directory: Path, files: Iterable[str]) -> list[Path]:
    keep = set(files)
    return [
        p
        for p in directory.glob("*.md")
        if p.name not in keep and p.read_text(encoding="utf-8").startswith(HEADER)
    ]
