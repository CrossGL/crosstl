"""The tracking issues of the translation matrix cells between CUDA, GLSL, HLSL and Metal."""

from itertools import product

from tools.coverage import issues

from .issue_cases import (
    backend_row,
    coverage_report,
    feature_entry,
    hlsl_row,
    issues_after,
    kinds,
)


def _cell(source, target, **outcomes):
    row = backend_row(
        source,
        "pair",
        [
            feature_entry(name, outcome, detail=f"{name} failed")
            for name, outcome in outcomes.items()
        ],
    )
    del row["language"], row["direction"], row["offered"]
    return {"source": source, "target": target, **row}


def _report(*cells, rows=()):
    return {**coverage_report(*rows), "pairs": list(cells)}


def test_every_cell_between_two_of_the_four_languages_gets_one_tracking_issue():
    languages = (*issues.PAIR_LANGUAGES, "wgsl")
    cells = [_cell(s, t, a="refused") for s, t in product(languages, repeat=2)]
    actions = issues.plan(_report(*cells), [])
    expected = {
        f"pair/{s}/{t}" for s, t in product(issues.PAIR_LANGUAGES, repeat=2) if s != t
    }
    assert len(expected) == 12
    assert kinds(actions) == sorted(("create", key) for key in expected)


def test_a_cell_issue_lists_its_gaps_and_links_the_cell_page():
    cell = _cell("hlsl", "metal", a="supported", b="wrong", c="refused", d="untested")
    (action,) = issues.plan(_report(cell), [])
    assert action.key == "pair/hlsl/metal"
    assert action.title == "HLSL to Metal translation coverage: 33.3 % (2 gaps)"
    assert action.labels == ("coverage", "backend:hlsl", "backend:metal")
    assert issues.marker_key(action.body) == "pair/hlsl/metal"
    assert "1 supported, 1 refused, 1 wrong and 1 untested features" in action.body
    assert "the Metal compiler accepts the output" in action.body
    assert "/blob/coverage/pairs/hlsl-to-metal.md" in action.body
    assert "- [ ] `b` wrong: b failed" in action.body
    assert "- [ ] `c` refused: c failed" in action.body
    assert "`a`" not in action.body and "`d`" not in action.body


def test_a_cell_issue_closes_when_its_gaps_are_fixed_and_reopens_on_regression():
    state = issues_after(issues.plan(_report(_cell("cuda", "glsl", a="wrong")), []))
    fixed = _report(_cell("cuda", "glsl", a="supported"))
    actions = issues.plan(fixed, state)
    assert [(a.kind, a.key, a.reason) for a in actions] == [
        ("close", "pair/cuda/glsl", "completed")
    ]
    assert actions[0].title == "CUDA to GLSL translation coverage: 100.0 % (0 gaps)"
    state = issues_after(actions, state)
    assert issues.plan(fixed, state) == []
    regressed = issues.plan(_report(_cell("cuda", "glsl", a="refused")), state)
    assert [(a.kind, a.key) for a in regressed] == [("reopen", "pair/cuda/glsl")]
    assert "CUDA to GLSL has 1 gap again" in regressed[0].comment


def test_a_report_without_the_matrix_or_an_unmeasured_cell_leaves_cell_issues_alone():
    state = issues_after(issues.plan(_report(_cell("glsl", "hlsl", a="wrong")), []))
    rows_only = coverage_report(hlsl_row(b="supported", c="supported"))
    assert not [a for a in issues.plan(rows_only, state) if a.key.startswith("pair/")]
    unmeasured = _report(_cell("glsl", "hlsl", a="untested"))
    assert unmeasured["pairs"][0]["score"] is None
    assert issues.plan(unmeasured, state) == []


def test_cell_issues_do_not_disturb_the_row_issues():
    report = _report(_cell("metal", "hlsl", a="refused"), rows=(hlsl_row(),))
    created = {a.key for a in issues.plan(report, []) if a.kind == "create"}
    assert {"pair/metal/hlsl", "tracking/hlsl/output"} <= created
    state = issues_after(issues.plan(report, []))
    assert not [a for a in issues.plan(report, state) if a.key.startswith("pair/")]
