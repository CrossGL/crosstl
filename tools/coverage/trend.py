"""The coverage trend: a hand-written SVG with one score line per language and direction, drawn from ``history.csv``.

Every line spans the time the history covers on a fixed 0 to 100 % scale, so rows compare honestly, and the change
since the row's first measured run is printed beside it. Rows the latest run no longer reports are left out. The
output depends only on the history text, so the same history always produces the same bytes.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from xml.sax.saxutils import escape

from . import chart, ir
from .chart import svg_text

ROW = 26
TOP = 74
HEIGHT = 18

STYLE = """\
text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif;font-size:13px}
.bg{fill:#ffffff;stroke:#d0d7de}.t{fill:#1f2328}.m{fill:#59636e}.track{fill:#eff2f5}
path{fill:none;stroke:currentColor;stroke-width:1.5}circle{fill:currentColor}
.output{color:#0969da}.input{color:#8250df}.ir{color:#1a7f37}
@media (prefers-color-scheme:dark){.bg{fill:#0d1117;stroke:#3d444d}.t{fill:#f0f6fc}.m{fill:#9198a1}\\
.track{fill:#262c36}.output{color:#4493f8}.input{color:#ab7df8}.ir{color:#3fb950}}"""

Point = tuple[str, float | None]
Key = tuple[str, str]


def _when(date: str) -> datetime:
    return datetime.fromisoformat(date)


def series(history: str) -> tuple[list[str], dict[Key, list[Point]]]:
    """The run dates of ``history`` in time order, and each row's ``(date, score)`` points.

    A row is keyed by ``(language, direction)``. Only rows the latest run reports are returned, in its order.
    """
    rows = [r for r in csv.DictReader(io.StringIO(history)) if r.get("date")]
    runs = sorted(
        {(r["date"], r["commit"]) for r in rows},
        key=lambda run: (_when(run[0]), run[1]),
    )
    if not runs:
        return [], {}
    position = {run: i for i, run in enumerate(runs)}
    rows.sort(key=lambda r: position[(r["date"], r["commit"])])
    latest = [
        (r["language"], r["direction"])
        for r in rows
        if (r["date"], r["commit"]) == runs[-1]
    ]
    points: dict[Key, list[Point]] = {key: [] for key in latest}
    for row in rows:
        key = (row["language"], row["direction"])
        if key in points:
            points[key].append(
                (row["date"], float(row["score"]) if row["score"] else None)
            )
    return [date for date, _ in runs], points


def change(points: list[Point]) -> str:
    """The latest score against the row's first measured one, or ``""`` when there is nothing to compare."""
    measured = [(date, score) for date, score in points if score is not None]
    if len(measured) < 2 or points[-1][1] is None:
        return ""
    delta = round(measured[-1][1] - measured[0][1], 1)
    since = measured[0][0][:10]
    return f"no change since {since}" if not delta else f"{delta:+.1f} since {since}"


def _score(points: list[Point]) -> str:
    latest = points[-1][1]
    return "n/a" if latest is None else f"{latest:.1f} %"


class _Axis:
    """Maps run dates and scores to coordinates inside one row's track."""

    def __init__(self, dates: list[str]) -> None:
        self.first = _when(dates[0])
        self.span = (_when(dates[-1]) - self.first).total_seconds()

    def x(self, date: str) -> float:
        if not self.span:
            return chart.BAR_X + chart.BAR_WIDTH
        offset = (_when(date) - self.first).total_seconds() / self.span
        return round(chart.BAR_X + chart.BAR_WIDTH * offset, 1)

    @staticmethod
    def y(score: float, base: float) -> float:
        return round(base - HEIGHT * score / 100, 1)


def _line(points: list[Point], base: float, axis: _Axis, css: str) -> list[str]:
    """The row's path, broken where a run has no score, with a dot on lone points and on the latest score."""
    segments: list[list[tuple[float, float]]] = [[]]
    for date, score in points:
        if score is None:
            segments.append([])
        else:
            segments[-1].append((axis.x(date), axis.y(score, base)))
    parts = []
    for segment in filter(None, segments):
        if len(segment) == 1:
            parts.append(
                f'<circle cx="{segment[0][0]:g}" cy="{segment[0][1]:g}" r="2"/>'
            )
            continue
        path = " ".join(
            f"{'L' if i else 'M'}{x:g} {y:g}" for i, (x, y) in enumerate(segment)
        )
        parts.append(f'<path d="{path}"/>')
    if points[-1][1] is not None and len(segments[-1]) > 1:
        x, y = segments[-1][-1]
        parts.append(f'<circle cx="{x:g}" cy="{y:g}" r="2"/>')
    return [f'<g class="{css}">', *parts, "</g>"]


def _rows(points: dict[Key, list[Point]], axis: _Axis) -> list[str]:
    body = []
    y = TOP
    previous = ""
    for (language, direction), row in points.items():
        expressiveness = language == ir.LANGUAGE
        if language != previous:
            body.append(svg_text(20, y, chart.name(language), "t"))
            previous = language
        body.append(
            svg_text(100, y, "expressive" if expressiveness else direction, "m")
        )
        base = y + 2
        body.append(
            f'<rect x="{chart.BAR_X}" y="{base - HEIGHT:g}" width="{chart.BAR_WIDTH}" height="{HEIGHT}" '
            'rx="2" class="track"/>'
        )
        body += _line(row, base, axis, "ir" if expressiveness else direction)
        body.append(svg_text(chart.BAR_X + chart.BAR_WIDTH + 10, y, _score(row), "t"))
        body.append(svg_text(chart.BAR_X + chart.BAR_WIDTH + 70, y, change(row), "m"))
        y += ROW
    return body


def _summary(points: dict[Key, list[Point]]) -> str:
    parts = []
    for (language, direction), row in points.items():
        text = f"{chart.name(language)} {'' if language == ir.LANGUAGE else direction + ' '}{_score(row)}"
        parts.append(f"{text} ({change(row)})" if change(row) else text)
    return "; ".join(parts) or "No published runs yet."


def render(history: str) -> str:
    """The trend chart for ``history``, the text of ``history.csv``, as an SVG document."""
    dates, points = series(history)
    height = TOP + ROW * max(len(points), 1) + 30
    body = [
        f'<rect x="0.5" y="0.5" width="{chart.WIDTH - 1}" height="{height - 1}" rx="6" class="bg"/>',
        svg_text(20, 30, "CrossTL coverage trend", "t"),
    ]
    if dates:
        runs = f"{len(dates)} run" + ("" if len(dates) == 1 else "s")
        body += [
            svg_text(chart.WIDTH - 20, 30, runs, "m", "end"),
            svg_text(chart.BAR_X, TOP - 26, dates[0][:10], "m"),
            svg_text(
                chart.BAR_X + chart.BAR_WIDTH, TOP - 26, dates[-1][:10], "m", "end"
            ),
            *_rows(points, _Axis(dates)),
        ]
    else:
        body.append(svg_text(20, TOP, "No published runs yet.", "m"))
    body.append(
        svg_text(
            20, height - 14, "Score per published run, on a 0 to 100 % scale.", "m"
        )
    )
    title = "CrossTL coverage over time per language and direction"
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart.WIDTH}" height="{height}" '
            f'viewBox="0 0 {chart.WIDTH} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(_summary(points))}</desc>',
            f"<style>{STYLE}</style>",
            *body,
            "</svg>",
            "",
        ]
    )
