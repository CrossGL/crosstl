"""The ``coverage.json`` report and its plain-text summary.

In memory every backend lists its features as dictionaries. ``coverage.json`` stores them as rows instead: a backend's
``columns`` names the fields of each row, and its ``details`` lists each diagnostic once, so the ``detail`` field
holds an index into it. That keeps the published file under a megabyte. :func:`load` and :func:`dumps` convert
between the two forms.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

from . import ir, upstream, validators
from .features import metal
from .model import DIRECTIONS, OUTCOMES, Result, Score

SCHEMA = 2
# The fields of a feature row in the order coverage.json stores them; any other field follows in sorted order.
COLUMNS = ("name", "category", "group", "outcome", "detail")
# Directions CrossTL does not offer at all; the report lists them so a reader can tell "not offered" from 0 %.
NOT_OFFERED = {
    ("opencl", "output"): "CrossTL has no OpenCL C generator",
    ("webgl", "input"): "CrossTL has no WebGL source frontend",
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


def build(results: list[Result], languages: list[str], directions: list[str]) -> dict:
    """Group ``results`` by language and direction and score each group."""
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
            score = Score()
            for item in group:
                score.add(item.outcome)
            backends.append(
                {
                    "language": language,
                    "direction": direction,
                    "offered": True,
                    "validator": validators.TOOLS[language],
                    "score": None if score.percent is None else round(score.percent, 1),
                    "counts": score.counts,
                    "features": [
                        {
                            "name": r.feature.name,
                            "category": r.feature.category,
                            "group": r.feature.group,
                            "outcome": r.outcome,
                            **({"detail": r.detail} if r.detail else {}),
                            **({"semantic": r.semantic} if r.semantic else {}),
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
        # Slang functions that SPIR-V lacks are compiled for HLSL and checked by DXC.
        | ({"dxc"} if "slang" in languages else set())
        | ({"spirv-dis"} if set(languages) & set(validators.SPIRV_COMPILERS) else set())
        # SPIR-V input probes are glslang's output.
        | (
            {"glslangValidator", "spirv-dis"}
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
    }


def merge(reports: list[dict]) -> dict:
    """One report from reports of separate runners, such as a Linux and a macOS job, with the IR row rebuilt.

    The runners must have measured the same commit; a language and direction measured twice is an error, because the
    merged score would depend on which runner came last.
    """
    if not reports:
        raise ValueError("no reports to merge")
    commits = {r["commit"] for r in reports}
    if len(commits) > 1:
        raise ValueError(
            f"reports come from different commits: {', '.join(sorted(commits))}"
        )
    rows: dict[tuple[str, str], dict] = {}
    for report in reports:
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
    backends = [
        rows[key] for key in sorted(rows, key=lambda k: (k[0], DIRECTIONS.index(k[1])))
    ]
    expressiveness = ir.build(backends)
    if expressiveness:
        backends.append(expressiveness)
    merged = dict(reports[0])
    merged["date"] = max(r["date"] for r in reports)
    merged["sources"] = dict(
        sorted({k: v for r in reports for k, v in r["sources"].items()}.items())
    )
    tools = sorted({tool for r in reports for tool in r["validators"]})
    merged["validators"] = {
        tool: next(
            (r["validators"][tool] for r in reports if r["validators"].get(tool)), None
        )
        for tool in tools
    }
    merged["backends"] = backends
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


def dumps(report: dict) -> str:
    """``report`` as the text of ``coverage.json``, with its features stored as rows."""
    stored = {
        **report,
        "backends": [_rows(b) if "features" in b else b for b in report["backends"]],
    }
    return json.dumps(stored, separators=(",", ":"), sort_keys=True) + "\n"


def loads(text: str) -> dict:
    """The report in ``text``, with each backend's features as dictionaries again."""
    data = json.loads(text)
    backends = [_features(b) if "columns" in b else b for b in data["backends"]]
    return {**data, "backends": backends}


def load(path: Path) -> dict:
    return loads(path.read_text(encoding="utf-8"))


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
    return "\n".join(lines)
