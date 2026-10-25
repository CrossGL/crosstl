import json

from tools.coverage import __main__ as cli
from tools.coverage import compare as compare_module
from tools.coverage import ir
from tools.coverage.compare import Baseline, baseline, checked, compare
from tools.coverage.publish import DETAIL_LIMIT


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
        "- HLSL output `mad`: supported → wrong (`dxc: error | bad line`)",
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


def test_a_reports_text_appears_only_in_code_spans():
    published = _report(_row("glsl", "output", {"sin": "supported"}))
    hostile = "`x` [link](https://example.com) @someone <img src=x>"
    current = _report(
        _row(
            "glsl",
            "output",
            {"sin": "wrong", hostile: "refused"},
            sin={"detail": hostile + "\n" + "y" * 400},
        )
    )
    changes = [
        line
        for line in compare(published, current).splitlines()
        if line.startswith("- ")
    ]
    code = "'x' [link](https://example.com) @someone <img src=x>"
    detail = changes[0].removeprefix(
        f"- GLSL output `sin`: supported → wrong (`{code} "
    )
    assert detail == "y" * (DETAIL_LIMIT - len(code) - 2) + "…`)"
    assert changes[1] == f"- GLSL output `{code}`: - → refused"
    for line in changes:
        assert all("@" not in part for part in line.split("`")[::2])


def test_malformed_rows_and_features_are_left_out():
    good = _row("glsl", "output", {"sin": "supported"})
    rows = [
        good,
        dict(good, language="<b>evil</b>"),
        dict(good, direction="sideways"),
        dict(good, language="hlsl", score="100"),
        dict(good, language="wgsl", score=float("nan")),
        dict(good, language="metal", score=True),
        dict(good, language="spirv", offered="yes"),
        dict(
            good,
            language="slang",
            features=[
                {"name": "a", "outcome": "fine"},
                {"name": 3, "outcome": "wrong"},
                {"name": "b", "outcome": "wrong", "detail": ["x"]},
                "c",
            ],
        ),
        "row",
    ]
    report = checked({"schema": 1, "commit": "c", "backends": rows})
    assert report == {
        "schema": 1,
        "backends": [
            {
                "language": "glsl",
                "direction": "output",
                "offered": True,
                "score": 100.0,
                "features": [{"name": "sin", "outcome": "supported"}],
            },
            {
                "language": "slang",
                "direction": "output",
                "offered": True,
                "score": 100.0,
                "features": [{"name": "b", "outcome": "wrong"}],
            },
        ],
    }
    assert checked({"backends": "none"}) == {"schema": None, "backends": []}


PUBLISHED, BASE, HEAD, MERGE = (c * 40 for c in "abcd")


def _commit(sha, subject):
    return {"sha": sha, "commit": {"message": subject + "\n\nbody (#1)"}}


def _api(ahead=(), behind=(), parents=(BASE, HEAD), fail=None):
    """A gh api stand-in for a main where the published coverage is ``ahead`` and ``behind`` commits away."""
    calls = []

    def api(path):
        calls.append(path)
        if fail:
            raise fail
        if path == f"repos/o/r/commits/{MERGE}":
            return {"sha": MERGE, "parents": [{"sha": sha} for sha in parents]}
        if path == f"repos/o/r/compare/{PUBLISHED}...{BASE}":
            return {"behind_by": len(behind), "commits": list(ahead)}
        if path == f"repos/o/r/compare/{BASE}...{PUBLISHED}":
            return {"behind_by": len(ahead), "commits": list(behind)}
        raise LookupError(path)

    api.calls = calls
    return api


def test_baseline_names_the_pull_requests_main_has_beyond_the_published_coverage():
    api = _api(
        ahead=[_commit("1" * 40, "Slang (#2967)"), _commit("2" * 40, "Mojo (#2971)")]
    )
    lag = baseline("o/r", PUBLISHED, MERGE, HEAD, api=api)
    assert lag == Baseline(ahead=("#2967", "#2971"))
    assert len(api.calls) == 2
    report = _report(_row("glsl", "output", {"sin": "supported"}))
    assert compare(report, report, baseline=lag).splitlines()[2] == (
        "The published coverage is 2 commits behind the `main` this change was measured on, so this comparison "
        "includes the changes of #2967 and #2971."
    )


def test_baseline_of_a_main_older_than_the_published_coverage():
    behind = [_commit(f"{n}" * 40, f"Change {n} (#{2930 + n})") for n in range(3)]
    lag = baseline("o/r", PUBLISHED, MERGE, HEAD, api=_api(behind=behind))
    assert lag == Baseline(behind=("#2930", "#2931", "#2932"))
    published = _report(_row("glsl", "output", {"sin": "supported"}))
    current = _report(_row("glsl", "output", {"sin": "refused"}))
    lines = compare(published, current, baseline=lag).splitlines()
    assert lines[
        lines.index("1 feature change outcome, 1 of them for the worse:") - 2
    ] == (
        "This change was measured on a `main` 3 commits behind the published coverage, so this comparison shows "
        "the changes of #2930, #2931 and #2932 reversed."
    )


def test_baseline_spells_commits_without_a_pull_request_by_hash_and_counts_the_rest():
    ahead = [_commit("e" * 40, "Direct commit")] + [
        _commit(f"{n % 10}" * 40, f"PR (#{n})") for n in range(11)
    ]
    lag = baseline("o/r", PUBLISHED, MERGE, HEAD, api=_api(ahead=ahead))
    assert lag.ahead[0] == "`eeeeeeeeeeee`"
    report = _report(_row("glsl", "output", {"sin": "supported"}))
    assert (
        compare(report, report, baseline=lag)
        .splitlines()[2]
        .endswith(
            "includes the changes of `eeeeeeeeeeee`, #0, #1, #2, #3, #4, #5, #6, #7, #8 and 2 more."
        )
    )


def test_baseline_of_the_published_main_adds_nothing():
    lag = baseline("o/r", BASE, MERGE, HEAD, api=_api())
    assert lag == Baseline()
    report = _report(_row("glsl", "output", {"sin": "supported"}))
    assert compare(report, report, baseline=lag) == compare(report, report)


def test_baseline_needs_a_merge_of_the_head(capsys):
    for parents in ((BASE,), (BASE, "e" * 40), (BASE, HEAD, "e" * 40)):
        assert (
            baseline("o/r", PUBLISHED, MERGE, HEAD, api=_api(parents=parents)) is None
        )
    api = _api()
    for repository, measured in (
        ("o/r/x", MERGE),
        ("o/r", "c"),
        ("o/r", MERGE.upper()),
    ):
        assert baseline(repository, PUBLISHED, measured, HEAD, api=api) is None
    assert api.calls == []
    assert (
        baseline("o/r", PUBLISHED, MERGE, HEAD, api=_api(fail=LookupError("HTTP 404")))
        is None
    )
    assert "warning: no baseline for dddddddddddd: HTTP 404" in capsys.readouterr().err


def test_command_names_the_pull_requests_in_between(tmp_path, capsys, monkeypatch):
    published = _report(_row("glsl", "output", {"sin": "supported"}), commit=PUBLISHED)
    current = _report(_row("glsl", "output", {"sin": "supported"}), commit=MERGE)
    paths = []
    for name, data in (("published.json", published), ("coverage.json", current)):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        paths.append(str(path))
    api = _api(ahead=[_commit("1" * 40, "Slang (#2967)")])
    monkeypatch.setattr(compare_module, "_gh", api)
    assert cli.main(["compare", *paths]) == 0
    assert "#2967" not in capsys.readouterr().out
    assert cli.main(["compare", *paths, "--repository", "o/r", "--head", HEAD]) == 0
    assert (
        "1 commit behind the `main` this change was measured on"
        in capsys.readouterr().out
    )
