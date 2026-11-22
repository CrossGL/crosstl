"""One documentation page per language, written from the published coverage report when the documentation is built.

The Sphinx extension ``docs/source/_ext/coverage_pages.py`` calls :func:`write` from its ``builder-inited`` hook. It
reads the report from a directory that holds the files of the ``coverage`` branch, or downloads ``coverage.json`` and
the feature files it names from :data:`PUBLISHED` with :func:`fetch`. The pages are Markdown, which the documentation
reads with MyST, and go to ``docs/source/languages/``, which git ignores, so nobody commits or regenerates them.

Each page has the language's rows of the report: the score and counts of each direction, the reference tool and the
runtime the probes ran on, the pinned feature lists, the measured features per category, the translation matrix cells
the language is the source or target of, and the gaps grouped by cause, as ``COVERAGE.md`` groups them.

The Coverage workflow publishes the same pages, in GitHub's Markdown (``github=True``), to ``languages/`` on the
``coverage`` branch beside ``COVERAGE.md`` (:func:`tools.coverage.publish.write_site`), so they are current even where
no documentation build has run since the last measurement.
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
    OUTCOMES,
    WORKFLOW,
    _bugs_url,
    _causes,
    _cell_score,
    _executed_cell,
    _untested,
    cell,
    label,
)
from .publish import page as cell_page

# Where the Coverage workflow publishes the report and its feature files.
PUBLISHED = "https://raw.githubusercontent.com/CrossGL/crosstl/coverage/"
COVERAGE_MD = "https://github.com/CrossGL/crosstl/blob/coverage/COVERAGE.md"
# The only names a page may have, so that none leads outside its directory. A downloaded file must have a name
# that report.load would read: ``report.FEATURE_FILE`` for a row, ``report.CELL_FILE`` for a matrix cell.
_LANGUAGE_ID = re.compile(r"[a-z0-9]+")
# The most bytes one downloaded file may have; the Coverage workflow publishes none over 1 MB.
_DOWNLOAD_LIMIT = 4_000_000
HEADER = "% Written from coverage.json by tools/coverage/docs.py when the documentation is built. Do not edit."
# The first line of a page published to the coverage branch, which GitHub renders without MyST.
GITHUB_HEADER = (
    "<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. "
    "Do not edit. -->"
)
DIRECTIONS = ("output", "input")
# The documentation sources of the checkout this module is in, where the backend guides are.
DOCS_SOURCE = Path(__file__).resolve().parents[2] / "docs" / "source"
# The backend guide of each language whose guide is not named after it.
GUIDE_NAMES = {"hlsl": "directx"}


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


def _pairs(data: dict, language: str, github: bool, repository: str) -> list[str]:
    """The measured translation matrix cells ``language`` is the source or the target of, each linked to its page."""
    cells = [
        c
        for c in data.get("pairs", [])
        if language in (c["source"], c["target"]) and c.get("features")
    ]
    if not cells:
        return []
    cells.sort(key=lambda c: (c["source"] != language, c["source"], c["target"]))
    stamp = data.get("matrix") or data
    base = "../" if github else f"https://github.com/{repository}/blob/coverage/"
    lines = [
        "## Translation pairs",
        "",
        f"The cells of the translation matrix that translate {chart.name(language)} straight to another language "
        "or another language straight to it, measured on "
        f"{_stamp({'commit': stamp.get('commit'), 'date': stamp.get('date')}, repository)}. Each cell measures "
        "the features of the source's input row that the target has too, and its page lists the gaps.",
        "",
        "| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for entry in cells:
        counts = entry["counts"]
        lines.append(
            f"| {chart.name(entry['source'])} | {chart.name(entry['target'])} | "
            f"[{_cell_score(entry)}]({base}{cell_page(entry)}) | "
            + " | ".join(str(counts.get(o, 0)) for o in OUTCOMES)
            + f" | {'yes' if entry.get('executes') else 'compile-only'} |"
        )
    return [*lines, ""]


def page(
    data: dict,
    language: str,
    guide: str | None = None,
    repository: str = "CrossGL/crosstl",
    github: bool = False,
) -> str:
    """The Markdown page of ``language``; ``guide`` is the document name of its backend guide, if it has one.

    ``github`` writes the page for the ``coverage`` branch: GitHub's Markdown, with links relative to it.
    """
    rows = _rows(data, language)
    if not rows:
        raise ValueError(f"the report has no row for {language!r}")
    name = chart.name(language)
    offered = [r for r in rows if r["offered"]]
    if github:
        built = "The Coverage workflow writes this page from `coverage.json` each time it publishes"
        coverage_md = "../COVERAGE.md"
    else:
        built = "The documentation build writes this page from the published `coverage.json`"
        coverage_md = COVERAGE_MD
    lines = [
        GITHUB_HEADER if github else HEADER,
        "",
        f"# {name}",
        "",
        f"Measured on {_stamp(data, repository)} by the "
        f"[Coverage workflow](https://github.com/{repository}/actions/workflows/{WORKFLOW}). {built}; "
        f"[COVERAGE.md]({coverage_md}) defines each outcome. Score = supported / (supported + refused + wrong).",
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
        link = (
            f"[backend guide](https://github.com/{repository}/blob/main/docs/source/{guide}.rst)"
            if github
            else f"{{doc}}`backend guide </{guide}>`"
        )
        lines += [
            f"The {link} describes how CrossTL translates {name}.",
            "",
        ]
    lines += _table(data, rows)
    lines += _sources(data, language, repository)
    lines += _categories(rows)
    lines += _pairs(data, language, github, repository)
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


def index(data: dict, repository: str = "CrossGL/crosstl", github: bool = False) -> str:
    """The page that lists every language with its scores and links to the language's page."""
    lines = [
        GITHUB_HEADER if github else HEADER,
        "",
        "# Coverage by language",
        "",
        "Each page lists what CrossTL translates to and from one language, measured on "
        f"{_stamp(data, repository)}. CI probes every feature of a pinned upstream feature list, checks each "
        "translation with the language's reference compiler or validator and, where it has a runtime for the "
        f"language, runs it. [COVERAGE.md]({'../COVERAGE.md' if github else COVERAGE_MD}) has the same numbers "
        "for every language on one page.",
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
    if github:
        return "\n".join(lines) + "\n"
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


def guides(srcdir: Path, names: Iterable[str]) -> dict[str, str]:
    """The document name of each language's backend guide under ``srcdir/backends/``, for those that have one."""
    found = {}
    for language in names:
        name = f"backends/{GUIDE_NAMES.get(language, language)}"
        if (srcdir / f"{name}.rst").is_file():
            found[language] = name
    return found


def pages(
    data: dict,
    guides: dict[str, str],
    repository: str = "CrossGL/crosstl",
    github: bool = False,
) -> dict[str, str]:
    """Every page by file name: ``index.md`` and one ``<language>.md`` per language of ``data``."""
    files = {"index.md": index(data, repository, github)}
    for language in languages(data):
        files[f"{language}.md"] = page(
            data, language, guides.get(language), repository, github
        )
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
