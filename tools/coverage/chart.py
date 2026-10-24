"""The coverage chart: a hand-written SVG of horizontal bars, one per language and direction.

The chart carries its own background, so it stays legible whatever page it is embedded in; a ``prefers-color-scheme``
rule switches it to dark colours for readers with a dark theme. The output depends only on the report, so the same
report always produces the same bytes.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from . import ir

NAMES = {
    "glsl": "GLSL",
    "hlsl": "HLSL",
    "metal": "Metal",
    "spirv": "SPIR-V",
    "webgl": "WebGL",
    "wgsl": "WGSL",
    "cuda": "CUDA",
    "hip": "HIP",
    "slang": "Slang",
    "rust": "Rust",
    "mojo": "Mojo",
    "opencl": "OpenCL C",
    ir.LANGUAGE: "CrossGL IR",
}

WIDTH = 780
ROW = 24
TOP = 58
BAR_X = 190
BAR_WIDTH = 340
BAR_HEIGHT = 12

STYLE = """\
text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif;font-size:13px}
.bg{fill:#ffffff;stroke:#d0d7de}.t{fill:#1f2328}.m{fill:#59636e}.rule{stroke:#d0d7de}.track{fill:#eff2f5}
.output{fill:#0969da}.input{fill:#8250df}.ir{fill:#1a7f37}
@media (prefers-color-scheme:dark){.bg{fill:#0d1117;stroke:#3d444d}.t{fill:#f0f6fc}.m{fill:#9198a1}\
.rule{stroke:#3d444d}.track{fill:#262c36}.output{fill:#4493f8}.input{fill:#ab7df8}.ir{fill:#3fb950}}"""


def name(language: str) -> str:
    return NAMES.get(language, language)


def measured(row: dict) -> int:
    counts = row["counts"]
    return counts["supported"] + counts["refused"] + counts["wrong"]


def percent(row: dict) -> str:
    """The score as printed everywhere: one decimal, or ``n/a`` when nothing was measured."""
    if not row.get("offered"):
        return "not offered"
    return "n/a" if row["score"] is None else f"{row['score']:.1f} %"


def svg_text(x: float, y: float, value: str, css: str, anchor: str = "") -> str:
    extra = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text x="{x:g}" y="{y:g}" class="{css}"{extra}>{escape(value)}</text>'


def _counts(row: dict) -> str:
    text = f"{row['counts']['supported']} of {measured(row)}"
    untested = row["counts"]["untested"]
    return f"{text}, {untested} untested" if untested else text


def _bar(row: dict, y: float) -> list[str]:
    top = y - BAR_HEIGHT + 2
    parts = [
        f'<rect x="{BAR_X}" y="{top:g}" width="{BAR_WIDTH}" height="{BAR_HEIGHT}" rx="2" class="track"/>'
    ]
    if not row.get("offered"):
        return [svg_text(BAR_X, y, "not offered: " + row.get("reason", ""), "m")]
    if row["score"]:
        width = round(BAR_WIDTH * row["score"] / 100, 1)
        css = "ir" if row["language"] == ir.LANGUAGE else row["direction"]
        parts.append(
            f'<rect x="{BAR_X}" y="{top:g}" width="{width:g}" height="{BAR_HEIGHT}" rx="2" class="{css}"/>'
        )
    parts.append(svg_text(BAR_X + BAR_WIDTH + 10, y, percent(row), "t"))
    parts.append(svg_text(BAR_X + BAR_WIDTH + 70, y, _counts(row), "m"))
    return parts


def _summary(rows: list[dict]) -> str:
    return "; ".join(
        f"{name(r['language'])} {'' if r['language'] == ir.LANGUAGE else r['direction'] + ' '}{percent(r)}"
        for r in rows
    )


def render(report: dict) -> str:
    """The chart for ``report`` as an SVG document."""
    rows = [r for r in report["backends"] if r["language"] != ir.LANGUAGE]
    expressiveness = [r for r in report["backends"] if r["language"] == ir.LANGUAGE]
    height = (
        TOP
        + ROW * (len(rows) + len(expressiveness))
        + (14 if expressiveness else 0)
        + 34
    )
    stamp = f"{report['commit'][:7] or 'uncommitted'} · {report['date'][:10]}"
    body = [
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="6" class="bg"/>',
        svg_text(20, 30, "CrossTL coverage", "t"),
        svg_text(WIDTH - 20, 30, stamp, "m", "end"),
    ]
    y = TOP
    previous = ""
    for row in rows:
        if row["language"] != previous:
            body.append(svg_text(20, y, name(row["language"]), "t"))
            previous = row["language"]
        body.append(svg_text(100, y, row["direction"], "m"))
        body += _bar(row, y)
        y += ROW
    if expressiveness:
        body.append(
            f'<line x1="20" y1="{y - 12}" x2="{WIDTH - 20}" y2="{y - 12}" class="rule"/>'
        )
        y += 14
        for row in expressiveness:
            body.append(svg_text(20, y, "CrossGL IR", "t"))
            body.append(svg_text(100, y, "expressive", "m"))
            body += _bar(row, y)
            y += ROW
    body.append(
        svg_text(
            20,
            height - 14,
            "Score = supported / (supported + refused + wrong), per generated probe "
            "validated by the reference tool.",
            "m",
        )
    )
    title = "CrossTL coverage per language and direction"
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(_summary(report["backends"]))}</desc>',
            f"<style>{STYLE}</style>",
            *body,
            "</svg>",
            "",
        ]
    )
