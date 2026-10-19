"""The ``coverage.json`` report and its plain-text summary."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

from . import ir, upstream, validators
from .model import OUTCOMES, Result, Score

SCHEMA = 1
# Directions CrossTL does not offer at all; the report lists them so a reader can tell "not offered" from 0 %.
NOT_OFFERED = {("wgsl", "input"): "CrossTL has no WGSL source frontend"}


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
                    "score": None if score.percent is None else round(score.percent, 1),
                    "counts": score.counts,
                    "features": [
                        {
                            "name": r.feature.name,
                            "category": r.feature.category,
                            "group": r.feature.group,
                            "outcome": r.outcome,
                            **({"detail": r.detail} if r.detail else {}),
                            **({"validator": r.validator} if r.validator else {}),
                            **({"semantic": r.semantic} if r.semantic else {}),
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
        | ({"spirv-dis"} if set(languages) & set(validators.SPIRV_COMPILERS) else set())
    )
    return {
        "schema": SCHEMA,
        "commit": _commit(),
        "date": (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "crosstl": _crosstl_version(),
        "sources": {
            language: {
                "repository": s.repository,
                "commit": s.commit,
                "path": s.path,
                "describes": s.describes,
            }
            for language, s in sorted(upstream.SOURCES.items())
            if language in languages
        },
        "validators": {tool: validators.version(tool) or None for tool in tools},
        "backends": backends,
    }


def write(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


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
