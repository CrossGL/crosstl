"""The ``coverage.json`` report and its plain-text summary.

In memory every backend lists its features as dictionaries. ``coverage.json`` stores them as rows instead: a backend's
``columns`` names the fields of each row, and its ``details`` lists each diagnostic once, so the ``detail`` field
holds an index into it. :func:`loads` and :func:`dumps` convert between the two forms.

The coverage branch splits the report further, so that no published file reaches a megabyte (ADR 0003):
:func:`site` leaves each backend's score and counts in ``coverage.json`` and moves its rows to the file named by its
``file`` field, ``features/<language>-<direction>.json``. :func:`load` reads either layout.

A feature that ran on a runtime names it in its ``runtime`` field, and a row counts its probes per runtime in
``executions``. A supported compute probe that a runtime did not run, or ran without a verdict, lists the reasons in
``unrun``, each starting with ``on <runtime>``. A report written with ``--execution-only`` (``"execution_only": true``) measures rows that another
report measures in full, only to run their probes on another runtime, as the Windows job runs the HLSL probes on
Direct3D 12 WARP. :func:`merge` keeps the full report's verdicts and adds the execution verdicts of such a report to
them (:func:`_with_executions`).
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from . import execution, ir, lowering, upstream, validators
from .features import metal
from .model import DIRECTIONS, OUTCOMES, PAIR, Result, Score

# 3: the coverage branch stores each backend's feature rows in its own file.
SCHEMA = 3
# The fields of a feature row in the order coverage.json stores them; any other field follows in sorted order.
COLUMNS = ("name", "category", "group", "outcome", "detail")
# The only names a ``file`` field may hold, so that loading a report never reads outside its directory.
FEATURE_FILE = re.compile(r"features/[a-z0-9]+-[a-z]+\.json")
# Likewise for a translation matrix cell (:mod:`.pairs`), whose features are in ``features/pairs/<source>-to-<target>``.
CELL_FILE = re.compile(r"features/pairs/[a-z0-9]+-to-[a-z0-9]+\.json")
# Directions CrossTL does not offer at all; the report lists them so a reader can tell "not offered" from 0 %.
NOT_OFFERED = {
    (
        "gles",
        "input",
    ): "GLSL ES sources are read as GLSL, which the GLSL and WebGL input rows measure",
    ("opencl", "output"): "CrossTL has no OpenCL C generator",
    ("wgsl", "input"): "CrossTL has no WGSL source frontend",
}


def _commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _crosstl_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("crosstl")
    except PackageNotFoundError:
        return ""


def _sources(languages: list[str]) -> dict[str, dict[str, str]]:
    sources = {
        language: {
            "repository": s.repository,
            "commit": s.commit,
            "path": s.path,
            "describes": s.describes,
        }
        for language, s in sorted(upstream.SOURCES.items())
        if language.split("-")[0] in languages
    }
    if "metal" in languages:
        sources["metal"] = metal.source(metal.LIST.read_text())
    return dict(sorted(sources.items()))


def _execution(executes: bool, group: list[Result], execute: bool) -> dict:
    """What a row or a cell records about running its probes.

    ``executes`` says whether the conformance harness can run the row's language, or the cell's probes and
    translations (:func:`.execution.runs_pair`), at all; a row without a runtime is compile-only. ``executed`` counts
    the probes that ran, 0 included, once ``execute`` asked for them to run, and ``runtime`` names the runtimes they
    ran on.
    """
    runtimes = Counter(r.runtime for r in group if r.runtime)
    executed = sum(runtimes.values())
    return {
        "executes": executes,
        **({"executed": executed} if executed or (execute and executes) else {}),
        **({"runtime": ", ".join(sorted(runtimes))} if runtimes else {}),
        **({"executions": dict(sorted(runtimes.items()))} if runtimes else {}),
    }


def _score(group: list[Result]) -> dict:
    score = Score()
    for item in group:
        score.add(item.outcome)
    return {
        "score": None if score.percent is None else round(score.percent, 1),
        "counts": score.counts,
    }


def _pairs(
    results: list[Result], cells: list[tuple[str, str]], execute: bool
) -> list[dict]:
    """The translation matrix cells ``cells`` as ``results`` measured them, in ``(source, target)`` order, each scored
    as a row is; a cell without a result is listed with no score."""
    grouped: dict[tuple[str, str], list[Result]] = {cell: [] for cell in cells}
    for result in results:
        if result.direction == PAIR:
            grouped.setdefault((result.feature.language, result.into), []).append(
                result
            )
    measured = []
    for (source, target), group in sorted(grouped.items()):
        group.sort(key=lambda r: r.feature.key)
        measured.append(
            {
                "source": source,
                "target": target,
                "validator": validators.TOOLS[target],
                **_execution(execution.runs_pair(source, target), group, execute),
                **_score(group),
                "features": [
                    {
                        "name": r.feature.name,
                        "category": r.feature.category,
                        "outcome": r.outcome,
                        **({"detail": r.detail} if r.detail else {}),
                    }
                    for r in group
                ],
            }
        )
    return measured


def build(
    results: list[Result],
    languages: list[str],
    directions: list[str],
    execute: bool = False,
    cells: list[tuple[str, str]] | None = None,
    execution_only: bool = False,
) -> dict:
    """Group ``results`` by language and direction and score each group.

    ``execute`` says that the executable probes of ``results`` were run, so a row records how many of them ran even
    when none did. With ``cells``, the report's ``pairs`` lists those cells of the translation matrix
    (:mod:`.pairs`), scored from the results of the pair direction. ``execution_only`` marks the report as one
    :func:`merge` reads only for its execution verdicts.
    """
    backends = []
    for language in languages:
        for direction in directions:
            if (language, direction) in NOT_OFFERED:
                backends.append(
                    {
                        "language": language,
                        "direction": direction,
                        "offered": False,
                        "reason": NOT_OFFERED[language, direction],
                    }
                )
                continue
            group = sorted(
                (
                    r
                    for r in results
                    if r.feature.language == language and r.direction == direction
                ),
                key=lambda r: r.feature.key,
            )
            if not group:
                continue
            backends.append(
                {
                    "language": language,
                    "direction": direction,
                    "offered": True,
                    "validator": validators.TOOLS[language],
                    **_execution(execution.runs(language), group, execute),
                    **_score(group),
                    "features": [
                        {
                            "name": r.feature.name,
                            "category": r.feature.category,
                            "group": r.feature.group,
                            "outcome": r.outcome,
                            **({"detail": r.detail} if r.detail else {}),
                            **({"runtime": r.runtime} if r.runtime else {}),
                            **({"unrun": [r.unrun]} if r.unrun else {}),
                            **({"semantic": r.semantic} if r.semantic else {}),
                            **({"expressed": r.expressed} if r.expressed else {}),
                            **(
                                {"crossgl": r.feature.crossgl}
                                if r.feature.crossgl not in ("", r.feature.name)
                                else {}
                            ),
                        }
                        for r in group
                    ],
                }
            )
    expressiveness = ir.build(backends)
    if expressiveness:
        backends.append(expressiveness)
    tools = sorted(
        {validators.TOOLS[language] for language in languages}
        | ({"spirv-as"} if "spirv" in languages else set())
        # Slang functions that SPIR-V lacks, or that slangc cannot write as valid SPIR-V, are compiled for HLSL and
        # checked by DXC.
        | ({"dxc"} if "slang" in languages else set())
        # HLSL features that need shader model 6.10 are checked by DXC's Shader Model 6.10 preview.
        | ({"dxc-preview"} if "hlsl" in languages else set())
        | ({"spirv-dis"} if set(languages) & set(lowering.SPIRV_COMPILERS) else set())
        # SPIR-V input probes are glslang's output, and DXC's for the constructs GLSL cannot spell.
        | (
            {"glslangValidator", "dxc", "spirv-dis"}
            if "spirv" in languages and "input" in directions
            else set()
        )
    )
    return {
        "schema": SCHEMA,
        "commit": _commit(),
        "date": (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "crosstl": _crosstl_version(),
        "sources": _sources(languages),
        "validators": {tool: validators.version(tool) or None for tool in tools},
        "backends": backends,
        **({} if cells is None else {"pairs": _pairs(results, cells, execute)}),
        **({"execution_only": True} if execution_only else {}),
    }


def _runtimes(feature: dict) -> set[str]:
    """The runtimes a feature's probes ran on."""
    return {name.strip() for name in feature.get("runtime", "").split(",")} - {""}


def _with_executions(row: dict, extra: dict) -> dict:
    """``row`` with the execution verdicts of ``extra``, an execution-only report's measurement of the same row.

    ``row`` keeps its static verdicts, so a probe ``extra`` judges differently without running it keeps its outcome.
    A probe that ran for ``extra`` records that runtime too, unless ``row`` found it refused, wrong without running it,
    or untested; one ``row`` found supported becomes wrong, with ``extra``'s diagnostic, when it diverged there. One
    ``row`` found supported that ``extra``'s runtime did not run adds ``extra``'s reasons to its ``unrun``.
    """
    measured = {(f["category"], f["name"]): f for f in extra["features"]}
    features = []
    for feature in row["features"]:
        other = measured.get((feature["category"], feature["name"]), {})
        if _runtimes(other) and (
            feature["outcome"] == "supported" or _runtimes(feature)
        ):
            runtimes = ", ".join(sorted(_runtimes(feature) | _runtimes(other)))
            feature = {**feature, "runtime": runtimes}
            if feature["outcome"] == "supported" and other["outcome"] == "wrong":
                feature.update(outcome="wrong", detail=other["detail"])
        elif other.get("unrun") and feature["outcome"] == "supported":
            feature = {**feature, "unrun": [*feature.get("unrun", []), *other["unrun"]]}
        features.append(feature)
    score = Score()
    for feature in features:
        score.add(feature["outcome"])
    counted = Counter(name for f in features for name in _runtimes(f))
    return {
        **row,
        "executed": sum(1 for f in features if _runtimes(f)),
        "runtime": ", ".join(sorted(counted)),
        "executions": dict(sorted(counted.items())),
        "score": None if score.percent is None else round(score.percent, 1),
        "counts": score.counts,
        "features": features,
    }


def merge(reports: list[dict]) -> dict:
    """One report from reports of separate runners, such as a Linux and a macOS job, with the IR row rebuilt.

    The runners must have measured the same commit; a language and direction, or a translation matrix cell, measured
    twice is an error, because the merged score would depend on which runner came last. An execution-only report only
    adds its execution verdicts to the rows another report measures (:func:`_with_executions`).
    """
    if not reports:
        raise ValueError("no reports to merge")
    commits = {r["commit"] for r in reports}
    if len(commits) > 1:
        raise ValueError(
            f"reports come from different commits: {', '.join(sorted(commits))}"
        )
    full = [r for r in reports if not r.get("execution_only")]
    if not full:
        raise ValueError("every report is execution-only")
    rows: dict[tuple[str, str], dict] = {}
    for report in full:
        for backend in report["backends"]:
            key = (backend["language"], backend["direction"])
            if backend["language"] == ir.LANGUAGE:
                continue
            if key in rows and rows[key]["offered"] and backend["offered"]:
                raise ValueError(
                    f"{key[0]} {key[1]} is measured by more than one report"
                )
            if key not in rows or backend["offered"]:
                rows[key] = backend
    for report in reports:
        if not report.get("execution_only"):
            continue
        for backend in report["backends"]:
            key = (backend["language"], backend["direction"])
            if backend["language"] == ir.LANGUAGE or not backend["offered"]:
                continue
            if key not in rows or not rows[key]["offered"]:
                raise ValueError(
                    f"{key[0]} {key[1]} has execution verdicts but no report measures it"
                )
            rows[key] = _with_executions(rows[key], backend)
    backends = [
        rows[key] for key in sorted(rows, key=lambda k: (k[0], DIRECTIONS.index(k[1])))
    ]
    expressiveness = ir.build(backends)
    if expressiveness:
        backends.append(expressiveness)
    merged = dict(full[0])
    merged["date"] = max(r["date"] for r in reports)
    merged["sources"] = dict(
        sorted({k: v for r in reports for k, v in r["sources"].items()}.items())
    )
    tools = sorted({tool for r in reports for tool in r["validators"]})
    # A full report names the tool version its verdicts come from.
    ordered = full + [r for r in reports if r.get("execution_only")]
    merged["validators"] = {
        tool: next(
            (r["validators"][tool] for r in ordered if r["validators"].get(tool)), None
        )
        for tool in tools
    }
    merged["backends"] = backends
    merged.pop("pairs", None)
    merged.pop("matrix", None)
    cells: dict[tuple[str, str], dict] = {}
    for report in reports:
        for cell in report.get("pairs", []):
            key = (cell["source"], cell["target"])
            if key in cells:
                raise ValueError(
                    f"{key[0]} to {key[1]} is measured by more than one report"
                )
            cells[key] = cell
    if any("pairs" in r for r in reports):
        merged["pairs"] = [cells[key] for key in sorted(cells)]
    return merged


def _rows(backend: dict) -> dict:
    features = backend["features"]
    fields = {name for feature in features for name in feature}
    columns = [c for c in COLUMNS if c in fields] + sorted(fields - set(COLUMNS))
    details = sorted({f["detail"] for f in features if f.get("detail")})
    index = {detail: number for number, detail in enumerate(details)}
    rows = [
        [index.get(f.get(c)) if c == "detail" else f.get(c) for c in columns]
        for f in features
    ]
    return {**backend, "columns": columns, "details": details, "features": rows}


def _features(backend: dict) -> dict:
    columns, details = backend["columns"], backend["details"]
    features = []
    for row in backend["features"]:
        feature = {c: v for c, v in zip(columns, row, strict=True) if v is not None}
        if "detail" in feature:
            feature["detail"] = details[feature["detail"]]
        features.append(feature)
    rest = {k: v for k, v in backend.items() if k not in ("columns", "details")}
    return {**rest, "features": features}


def _json(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n"


def dumps(report: dict) -> str:
    """``report`` as the text of ``coverage.json``, with its features stored as rows."""
    stored = {
        **report,
        "backends": [_rows(b) if "features" in b else b for b in report["backends"]],
        **({"pairs": [_rows(c) for c in report["pairs"]]} if "pairs" in report else {}),
    }
    return _json(stored)


def cell_file(cell: dict) -> str:
    """The path, on the coverage branch, of the features of the translation matrix cell ``cell``."""
    return f"features/pairs/{cell['source']}-to-{cell['target']}.json"


def site(report: dict) -> dict[str, str]:
    """``report`` as the files of the coverage branch, by path: ``coverage.json`` without any feature rows, one
    ``features/<language>-<direction>.json`` with the rows of each backend that has features, and one
    :func:`cell_file` with those of each cell of the translation matrix.
    """
    files = {}
    backends = []
    for backend in report["backends"]:
        if "features" not in backend:
            backends.append(backend)
            continue
        name = f"features/{backend['language']}-{backend['direction']}.json"
        if not FEATURE_FILE.fullmatch(name):
            raise ValueError(f"{name!r} is not the name of a feature file")
        stored = _rows(backend)
        files[name] = _json(
            {
                key: stored[key]
                for key in ("language", "direction", "columns", "details", "features")
            }
        )
        rest = {k: v for k, v in backend.items() if k != "features"}
        backends.append({**rest, "file": name})
    cells = []
    for cell in report.get("pairs", []):
        name = cell_file(cell)
        if not CELL_FILE.fullmatch(name):
            raise ValueError(f"{name!r} is not the name of a cell file")
        stored = _rows(cell)
        files[name] = _json(
            {
                key: stored[key]
                for key in ("source", "target", "columns", "details", "features")
            }
        )
        rest = {k: v for k, v in cell.items() if k != "features"}
        cells.append({**rest, "file": name})
    published = {**report, "backends": backends}
    if "pairs" in report:
        published["pairs"] = cells
    return {"coverage.json": _json(published), **files}


def _linked(
    entry: dict,
    directory: Path,
    pattern: re.Pattern = FEATURE_FILE,
    owners: tuple[str, str] = ("language", "direction"),
) -> dict:
    """``entry``, a backend or a matrix cell, with the rows of the feature file it names, read from ``directory``.

    The file must match ``pattern`` and name the entry's ``owners`` fields as its own.
    """
    name = entry["file"]
    if not isinstance(name, str) or not pattern.fullmatch(name):
        raise ValueError(f"{name!r} is not the name of a feature file")
    base = directory.resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"{name} leads outside {directory}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    owner = tuple(stored.get(field) for field in owners)
    if owner != tuple(entry.get(field) for field in owners):
        raise ValueError(f"{name} holds the features of {owner[0]} {owner[1]}")
    rest = {k: v for k, v in entry.items() if k != "file"}
    return {**rest, **{k: stored[k] for k in ("columns", "details", "features")}}


def _expanded(data: dict) -> dict:
    backends = [
        _features(b) if isinstance(b, dict) and "columns" in b else b
        for b in data["backends"]
    ]
    cells = [_features(c) if "columns" in c else c for c in data.get("pairs", [])]
    return {
        **data,
        "backends": backends,
        **({"pairs": cells} if "pairs" in data else {}),
    }


def loads(text: str) -> dict:
    """The report in ``text``, with each backend's features as dictionaries again."""
    return _expanded(json.loads(text))


def load(path: Path) -> dict:
    """The report in ``path``, reading the feature file each backend or matrix cell names, if any, from the same
    directory."""
    data = json.loads(path.read_text(encoding="utf-8"))
    backends = [
        _linked(b, path.parent) if isinstance(b, dict) and "file" in b else b
        for b in data["backends"]
    ]
    linked = {**data, "backends": backends}
    if "pairs" in data:
        linked["pairs"] = [
            (
                _linked(c, path.parent, CELL_FILE, ("source", "target"))
                if isinstance(c, dict) and "file" in c
                else c
            )
            for c in data["pairs"]
        ]
    return _expanded(linked)


def write(report: dict, path: Path) -> None:
    path.write_text(dumps(report), encoding="utf-8")


def summary(report: dict) -> str:
    """One line per language and direction, for terminals and CI logs."""
    lines = [
        f"{'backend':<16}{'score':>7}  " + "  ".join(f"{name:>9}" for name in OUTCOMES)
    ]
    for backend in report["backends"]:
        label = (
            "CrossGL IR"
            if backend["language"] == ir.LANGUAGE
            else f"{backend['language']} {backend['direction']}"
        )
        if not backend["offered"]:
            lines.append(f"{label:<16}{'-':>7}  not offered: {backend['reason']}")
            continue
        score = "n/a" if backend["score"] is None else f"{backend['score']:.1f}%"
        lines.append(
            f"{label:<16}{score:>7}  "
            + "  ".join(f"{backend['counts'][o]:>9}" for o in OUTCOMES)
        )
    for cell in report.get("pairs", []):
        label = f"{cell['source']} to {cell['target']}"
        score = "n/a" if cell["score"] is None else f"{cell['score']:.1f}%"
        lines.append(
            f"{label:<16}{score:>7}  "
            + "  ".join(f"{cell['counts'][o]:>9}" for o in OUTCOMES)
        )
    return "\n".join(lines)
