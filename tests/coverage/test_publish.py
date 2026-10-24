import csv
import io
import json
import xml.etree.ElementTree as ElementTree
from urllib.parse import unquote

from tools.coverage import __main__ as cli
from tools.coverage import chart, issues, publish
from tools.coverage.causes import grouped

SVG = "{http://www.w3.org/2000/svg}"


def _row(language, direction, **counts):
    counts = {"supported": 0, "refused": 0, "wrong": 0, "untested": 0, **counts}
    measured = counts["supported"] + counts["refused"] + counts["wrong"]
    features = [
        {
            "name": f"f{outcome}{i}",
            "category": "function",
            "group": "g",
            "outcome": outcome,
        }
        | (
            {"detail": f"{outcome} | detail\nsecond line"}
            if outcome in ("refused", "wrong")
            else {}
        )
        for outcome, n in counts.items()
        for i in range(n)
    ]
    return {
        "language": language,
        "direction": direction,
        "offered": True,
        "score": round(100 * counts["supported"] / measured, 1) if measured else None,
        "counts": counts,
        "features": features,
    }


def _report(commit="0123456789abcdef", date="2026-10-19T03:41:00Z"):
    return {
        "schema": 1,
        "commit": commit,
        "date": date,
        "crosstl": "1.2.3",
        "sources": {
            "glsl": {
                "repository": "KhronosGroup/GLSL",
                "commit": "0099bf83b0288469917fc2e2972cb0a2b74d6711",
                "path": "chapters/builtinfunctions.adoc",
                "describes": "GLSL built-in functions",
            },
            "metal": {
                "path": "tools/coverage/features/metal.txt",
                "describes": "MSL library",
            },
            "glsl-extensions": {
                "repository": "KhronosGroup/GLSL",
                "commit": "0099bf83b0288469917fc2e2972cb0a2b74d6711",
                "path": "extensions/ext/GL_EXT_shader_atomic_float.txt",
                "describes": "GLSL atomic float extension",
            },
        },
        "validators": {"glslangValidator": "16.0", "metal": None},
        "backends": [
            _row("glsl", "output", supported=3, wrong=1, untested=2),
            _row("glsl", "input", supported=4),
            {
                "language": "wgsl",
                "direction": "input",
                "offered": False,
                "reason": "CrossTL has no WGSL source frontend",
            },
            _row("crossgl", "ir", supported=2, untested=5),
        ],
    }


def test_chart_is_valid_svg_with_scores_commit_and_date():
    svg = chart.render(_report())
    root = ElementTree.fromstring(svg)
    texts = [t.text for t in root.iter(f"{SVG}text")]
    assert "0123456 · 2026-10-19" in texts
    assert "75.0 %" in texts and "100.0 %" in texts
    assert "3 of 4, 2 untested" in texts
    assert "not offered: CrossTL has no WGSL source frontend" in texts
    assert "CrossGL IR" in texts
    assert root.find(f"{SVG}title").text.startswith("CrossTL coverage")
    # The supported share of the bar, in the direction's colour, is proportional to the score.
    bars = [r for r in root.iter(f"{SVG}rect") if r.get("class") == "output"]
    assert [float(r.get("width")) for r in bars] == [chart.BAR_WIDTH * 0.75]


def test_chart_has_no_gradients_or_scripts_and_is_deterministic():
    svg = chart.render(_report())
    assert svg == chart.render(_report())
    for banned in ("Gradient", "<script", "<filter", "<image", "href="):
        assert banned not in svg
    # Both themes are styled, and the chart draws its own background.
    assert "prefers-color-scheme:dark" in svg and 'class="bg"' in svg


def test_chart_shows_unmeasured_rows_without_a_bar():
    data = _report()
    data["backends"] = [_row("hlsl", "output", untested=3)]
    root = ElementTree.fromstring(chart.render(data))
    assert "n/a" in [t.text for t in root.iter(f"{SVG}text")]
    assert not [r for r in root.iter(f"{SVG}rect") if r.get("class") == "output"]


def test_markdown_lists_scores_sources_and_gaps():
    text = publish.markdown(_report())
    assert "| GLSL | output | 75.0 % | 3 | 0 | 1 | 2 |" in text
    assert "| WGSL | input | not offered | | | | |" in text
    assert "| CrossGL IR | expressiveness | 100.0 % | 2 | 0 | 0 | 5 |" in text
    assert "https://github.com/CrossGL/crosstl/commit/0123456789abcdef" in text
    assert (
        "KhronosGroup/GLSL/blob/0099bf83b0288469917fc2e2972cb0a2b74d6711/chapters"
        in text
    )
    assert "crosstl/blob/0123456789abcdef/tools/coverage/features/metal.txt" in text
    assert "metal (not installed)" in text
    # A second list for a language is listed under the language's name, not its key.
    assert "| GLSL | [GLSL atomic float extension](" in text
    assert "glsl-extensions" not in text
    # A gap's diagnostic is its first line, in a code span, followed by the features that give it.
    assert "- `wrong | detail`: `fwrong0`" in text
    assert "<summary>2 untested: function 2</summary>" in text
    # The trend chart is published next to COVERAGE.md and embedded with a relative link.
    assert "](trend.svg)" in text


def test_markdown_counts_measured_features_per_category():
    data = _report()
    data["backends"][0]["features"][4]["category"] = "stage"
    data["backends"][1]["features"][0]["category"] = "stage"
    data["backends"][3]["features"][3]["category"] = "method"
    lines = publish.markdown(data).splitlines()
    # "method" has only an untested feature, so it gets no column.
    assert "| Language | Direction | function | stage |" in lines
    # A category whose features are all untested gets a blank cell rather than 0/0.
    assert "| GLSL | output | 3/4 | |" in lines
    assert "| GLSL | input | 3/3 | 1/1 |" in lines
    assert "| CrossGL IR | expressiveness | 2/2 | |" in lines
    # Rows that are not offered have no features and stay out of this table.
    assert sum(line.startswith("| WGSL | input |") for line in lines) == 1


def test_markdown_truncates_long_details():
    data = _report()
    data["backends"][0]["features"][3]["detail"] = "x" * 500
    line = next(
        line for line in publish.markdown(data).splitlines() if "`fwrong0`" in line
    )
    assert "…`: `fwrong0`" in line and len(line) < publish.CAUSE_LIMIT + 40


def _gap(name, outcome, detail, category="function"):
    return {
        "name": name,
        "category": category,
        "group": "g",
        "outcome": outcome,
        "detail": detail,
    }


def test_markdown_groups_gaps_by_cause_as_the_bug_issues_do():
    data = _report()
    row = data["backends"][0]
    row["features"] = [
        _gap("abs", "wrong", 'probe.cu(12): error: identifier "outp" is undefined'),
        _gap("acos", "wrong", 'probe.cu(14): error: identifier "outp" is undefined'),
        _gap(
            "vertex",
            "wrong",
            'probe.cu(9): error: identifier "outp" is undefined',
            "stage",
        ),
        _gap("ldexp", "wrong", "probe.cu(3): error: no instance of ldexp matches"),
        _gap("fma", "refused", "CrossGL has no fused multiply-add"),
        _gap("fmaf", "refused", "CrossGL has no fused multiply-add"),
    ]
    row["counts"] = {"supported": 0, "refused": 2, "wrong": 4, "untested": 0}
    lines = publish.markdown(data).splitlines()
    # One item per cause, across categories, with the probe's variables kept visible in the code span.
    assert '- `identifier "<probe>" is undefined` (3): `abs`, `acos`, `vertex`' in lines
    # A cause of one feature shows that feature's own diagnostic, which names it.
    assert "- `no instance of ldexp matches`: `ldexp`" in lines
    assert "- `CrossGL has no fused multiply-add` (2): `fma`, `fmaf`" in lines
    assert lines.index("Refused, by diagnostic:") > lines.index(
        "- `no instance of ldexp matches`: `ldexp`"
    )
    # The causes are the bug issues' causes, and the link finds the issues by the labels issues.py gives them.
    assert [len(f) for _, f in grouped(row)] == [3, 1]
    link = next(line for line in lines if line.startswith("Wrong, by cause"))
    query = unquote(link.split("issues?q=", 1)[1].split(")", 1)[0])
    labels = {
        word.removeprefix("label:")
        for word in query.split()
        if word.startswith("label:")
    }
    assert labels == set(issues.bug_labels(row)) - {issues.AREA[row["direction"]]}


def test_markdown_ir_gaps_keep_their_language_and_link_no_bugs():
    data = _report()
    row = data["backends"][3]
    row["features"] = [
        _gap(
            "cuda/abs",
            "wrong",
            'cuda: probe.cu(12): error: identifier "outp" is undefined',
        ),
        _gap(
            "cuda/acos",
            "wrong",
            'cuda: probe.cu(13): error: identifier "outp" is undefined',
        ),
    ]
    row["counts"] = {"supported": 0, "refused": 0, "wrong": 2, "untested": 0}
    lines = publish.markdown(data).splitlines()
    assert (
        '- `cuda: identifier "<probe>" is undefined` (2): `cuda/abs`, `cuda/acos`'
        in lines
    )
    assert "Wrong, by cause:" in lines


def test_markdown_groups_untested_features_by_reason():
    data = _report()
    row = data["backends"][0]
    row["features"] = [
        _gap("ClusterID", "untested", "needs shader model 6.10"),
        _gap("printf", "untested", "DXC compiles printf only to SPIR-V\nsecond line"),
        _gap("Sample", "untested", "needs shader model 6.10", "method"),
        _gap("abs", "wrong", "probe.hlsl:3:1: error: no matching function"),
    ]
    row["counts"] = {"supported": 0, "refused": 0, "wrong": 1, "untested": 3}
    lines = publish.markdown(data).splitlines()
    start = lines.index("### GLSL output") + 1
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("### "))
    section = lines[start:end]
    # The summary keeps the count per category; the list groups by reason, the most shared first.
    summary = section.index("<summary>3 untested: function 2, method 1</summary>")
    # A reason keeps its numbers, which here name a shader model rather than a probe's line, and shows its first line.
    assert section[summary + 1 : summary + 6] == [
        "",
        "- `needs shader model 6.10` (2): `ClusterID`, `Sample`",
        "- `DXC compiles printf only to SPIR-V`: `printf`",
        "",
        "</details>",
    ]
    # The untested list sits in its own collapsed block, after the gaps'.
    wrong = next(i for i, line in enumerate(section) if line.startswith("Wrong, by"))
    assert section.count("<details>") == 2 and wrong < summary


def _history_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_history_appends_runs_replaces_reruns_and_stays_bounded(monkeypatch):
    first = publish.history(_report())
    assert [r["language"] for r in _history_rows(first)] == ["glsl", "glsl", "crossgl"]
    assert _history_rows(first)[0]["score"] == "75.0"
    # Publishing the same run twice does not duplicate it.
    assert publish.history(_report(), first) == first
    second = publish.history(
        _report(commit="fedcba", date="2026-10-20T03:41:00Z"), first
    )
    assert len(_history_rows(second)) == 6
    monkeypatch.setattr(publish, "HISTORY_RUNS", 1)
    third = publish.history(_report(commit="aaaa", date="2026-10-21T03:41:00Z"), second)
    assert {r["commit"] for r in _history_rows(third)} == {"aaaa"}


def test_publish_command_writes_the_five_files(tmp_path):
    report_path = tmp_path / "coverage.json"
    report_path.write_text(json.dumps(_report()))
    old = tmp_path / "old.csv"
    old.write_text(publish.history(_report(commit="old", date="2026-10-01T00:00:00Z")))
    site = tmp_path / "site"
    assert (
        cli.main(
            [
                "publish",
                str(report_path),
                "--output-dir",
                str(site),
                "--history",
                str(old),
            ]
        )
        == 0
    )
    assert sorted(p.name for p in site.iterdir()) == [
        "COVERAGE.md",
        "coverage.json",
        "coverage.svg",
        "history.csv",
        "trend.svg",
    ]
    assert {r["commit"] for r in _history_rows((site / "history.csv").read_text())} == {
        "old",
        "0123456789abcdef",
    }
    # The trend is drawn from the history just written, so it includes this run.
    assert ">2 runs<" in (site / "trend.svg").read_text()
    # A missing history file is a first publication, not an error.
    assert (
        cli.main(
            [
                "publish",
                str(report_path),
                "--output-dir",
                str(site),
                "--history",
                str(tmp_path / "none"),
            ]
        )
        == 0
    )
