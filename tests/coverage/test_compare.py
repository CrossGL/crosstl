import json

from tools.coverage import __main__ as cli
from tools.coverage import ir
from tools.coverage.compare import compare


def _row(language, direction, outcomes, **extra):
    features = [
        {"name": name, "category": "function", "group": "g", "outcome": outcome}
        | extra.get(name, {})
        for name, outcome in outcomes.items()
    ]
    counts = dict.fromkeys(("supported", "refused", "wrong", "untested"), 0)
    for feature in features:
        counts[feature["outcome"]] += 1
    measured = counts["supported"] + counts["refused"] + counts["wrong"]
    return {
        "language": language,
        "direction": direction,
        "offered": True,
        "score": round(100 * counts["supported"] / measured, 1) if measured else None,
        "counts": counts,
        "features": features,
    }


def _report(*rows, commit="0123456789abcdef0123"):
    backends = list(rows)
    expressiveness = ir.build(backends)
    if expressiveness:
        backends.append(expressiveness)
    return {
        "schema": 1,
        "commit": commit,
        "date": "2026-10-19T03:41:00Z",
        "backends": (
            backends
            + [
                {
                    "language": "wgsl",
                    "direction": "input",
                    "offered": False,
                    "reason": "none",
                }
            ]
        ),
    }


def test_unchanged_report_says_so():
    report = _report(_row("glsl", "output", {"sin": "supported", "fma": "refused"}))
    text = compare(report, report)
    assert text == (
        "Against the coverage published for `0123456789ab` on 2026-10-19, coverage is unchanged.\n"
    )


def test_score_changes_and_feature_changes_list_regressions_first():
    published = _report(
        _row("glsl", "output", {"sin": "supported", "fma": "refused", "cos": "wrong"}),
        _row("hlsl", "output", {"mad": "supported", "abs": "supported"}),
    )
    current = _report(
        _row(
            "glsl",
            "output",
            {"sin": "supported", "fma": "supported", "cos": "refused"},
        ),
        _row(
            "hlsl",
            "output",
            {"mad": "wrong", "abs": "supported"},
            mad={"detail": "dxc: error | bad\nline"},
        ),
    )
    lines = compare(published, current).splitlines()
    assert "| GLSL output | 33.3 % | 66.7 % (+33.4) |" in lines
    assert "| HLSL output | 100.0 % | 50.0 % (-50.0) |" in lines
    assert "3 features change outcome, 1 of them for the worse:" in lines
    changes = [line for line in lines if line.startswith("- ")]
    assert changes == [
        "- HLSL output `mad`: supported → wrong (dxc: error \\| bad line)",
        "- GLSL output `fma`: refused → supported",
        "- GLSL output `cos`: wrong → refused",
    ]


def test_skipped_leg_is_named_and_not_an_ir_regression():
    glsl = _row("glsl", "output", {"sin": "supported"})
    metal = _row("metal", "output", {"simd_sum": "supported"})
    published = _report(glsl, metal)
    current = _report(glsl)
    text = compare(published, current)
    assert "CrossGL IR" not in text.split("Not measured")[0]
    assert "coverage is unchanged" in text
    assert text.endswith("Not measured in this run: Metal output.\n")


def test_new_rows_and_features_are_marked():
    published = _report(_row("glsl", "output", {"sin": "supported"}))
    current = _report(
        _row("glsl", "output", {"sin": "supported", "cos": "refused"}),
        _row("rust", "output", {"sin": "supported"}),
    )
    lines = compare(published, current).splitlines()
    assert "| Rust output | new | 100.0 % |" in lines
    assert [line for line in lines if line.startswith("- ")] == [
        "- GLSL output `cos`: - → refused"
    ]


def test_untested_features_are_not_regressions():
    published = _report(_row("glsl", "output", {"sin": "supported", "cos": "untested"}))
    current = _report(_row("glsl", "output", {"sin": "untested", "cos": "wrong"}))
    lines = compare(published, current).splitlines()
    assert "2 features change outcome:" in lines


def test_long_lists_are_cut():
    names = {f"f{i:02}": "supported" for i in range(50)}
    published = _report(_row("glsl", "output", names))
    current = _report(_row("glsl", "output", dict.fromkeys(names, "refused")))
    lines = compare(published, current, limit=5).splitlines()
    assert sum(line.startswith("- ") for line in lines) == 6
    assert lines[-1] == "- and 45 more."


def test_schema_change_is_not_compared():
    published = _report(_row("glsl", "output", {"sin": "supported"}))
    current = dict(published, schema=2)
    assert "another schema" in compare(published, current)


def test_command_prints_the_comparison(tmp_path, capsys):
    published = _report(_row("glsl", "output", {"sin": "supported"}))
    current = _report(_row("glsl", "output", {"sin": "refused"}))
    paths = []
    for name, data in (("published.json", published), ("coverage.json", current)):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        paths.append(str(path))
    assert cli.main(["compare", *paths]) == 0
    out = capsys.readouterr().out
    assert "| GLSL output | 100.0 % | 0.0 % (-100.0) |" in out
    assert "- GLSL output `sin`: supported → refused" in out
