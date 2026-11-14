"""How reconciliation files, updates, closes and reopens the tracking and bug issues."""

from dataclasses import replace

from tools.coverage import issues

from .issue_cases import (
    Repository,
    backend_row,
    bug_for,
    coverage_report,
    detail_for,
    feature_entry,
    hlsl_row,
    issues_after,
    kinds,
)


def test_keys_are_marker_safe_and_round_trip():
    key = issues.key("wrong", "hlsl", "output", "Texture2D<float4>::Sample x")
    assert key == "wrong/hlsl/output/Texture2D_float4_::Sample_x"
    assert issues.marker_key("text\n" + issues.MARKER.format(key) + "\nmore") == key
    assert issues.marker_key("an ordinary issue") == ""
    assert issues.marker_key(None) == ""


def test_first_run_files_tracking_and_bug_issues():
    report = coverage_report(
        hlsl_row(), backend_row("glsl", "output", [feature_entry("x", "supported")])
    )
    actions = issues.plan(report, [])
    assert kinds(actions) == [
        ("create", "tracking/hlsl/output"),
        ("create", bug_for("hlsl", "b")),
    ]
    tracking = actions[0]
    assert tracking.title == "HLSL output coverage: 33.3 % (2 gaps)"
    assert tracking.labels == (
        "coverage",
        "backend:hlsl",
        "direction:output",
        "area:codegen",
    )
    assert tracking.body.startswith(issues.MARKER.format("tracking/hlsl/output") + "\n")
    assert "- [ ] `b` wrong: B failed" in tracking.body
    assert "- [ ] `c` refused: C failed" in tracking.body
    assert "`d`" not in tracking.body
    bug = actions[1]
    assert bug.title == "HLSL output: B failed"
    assert (
        "The coverage probe for `b` (HLSL output, g) gives a wrong result." in bug.body
    )
    assert bug.labels[0] == "bug"
    assert (
        "python -m tools.coverage run --backend hlsl --direction output --feature b --keep probes"
        in bug.body
    )
    assert "`probes/hlsl-output/b/` then holds the probe" in bug.body


def test_a_run_links_the_bugs_it_files_and_the_next_run_changes_nothing():
    report = coverage_report(hlsl_row())
    repository = Repository()
    logged = []
    actions = issues.reconcile(report, repository, log=logged.append)
    assert kinds(actions) == [
        ("create", "tracking/hlsl/output"),
        ("create", bug_for("hlsl", "b")),
        ("edit", "tracking/hlsl/output"),
    ]
    assert len(logged) == 3
    bug = next(i for i in repository.issues if i.key == bug_for("hlsl", "b"))
    tracking = next(i for i in repository.issues if i.key == "tracking/hlsl/output")
    assert f"- [ ] `b` wrong (#{bug.number}): B failed" in tracking.body
    assert issues.reconcile(report, repository, log=logged.append) == []
    assert len(logged) == 3


class LaggingRepository(Repository):
    """A repository whose issue list leaves out the issues created since the last :meth:`settle`, as GitHub's can."""

    def __init__(self, existing=()):
        super().__init__(existing)
        self.fresh = set()

    def managed(self):
        return [i for i in self.issues if i.number not in self.fresh]

    def apply(self, action):
        number = super().apply(action)
        if action.kind == "create":
            self.fresh.add(number)
        return number

    def settle(self):
        self.fresh.clear()


def test_a_run_links_its_new_bugs_when_the_issue_list_lags():
    report = coverage_report(hlsl_row())
    repository = LaggingRepository()
    actions = issues.reconcile(report, repository, log=lambda line: None)
    assert kinds(actions) == [
        ("create", "tracking/hlsl/output"),
        ("create", bug_for("hlsl", "b")),
        ("edit", "tracking/hlsl/output"),
    ]
    assert len(repository.issues) == 2
    bug = next(i for i in repository.issues if i.key == bug_for("hlsl", "b"))
    tracking = next(i for i in repository.issues if i.key == "tracking/hlsl/output")
    assert f"- [ ] `b` wrong (#{bug.number}): B failed" in tracking.body
    repository.settle()
    assert issues.reconcile(report, repository, log=lambda line: None) == []


def test_applied_issues_match_what_github_keeps():
    old = issues.Issue(7, "tracking/hlsl/output", "open", "t", "b", ("coverage",))
    edit = issues.Action("edit", "tracking/hlsl/output", "t2", "b2", ("bug",), 7)
    close = issues.Action("close", "k", "t3", "b3", (), 9, "gone", "completed")
    create = issues.Action("create", "wrong/x", "new", "body", ("coverage", "bug"))
    gone = issues.Issue(9, "k", "open", "t", "b")
    after = issues.applied([old, gone], [(edit, 7), (close, 9), (create, 12)])
    assert after == [
        issues.Issue(
            7, "tracking/hlsl/output", "open", "t2", "b2", ("coverage", "bug")
        ),
        issues.Issue(9, "k", "closed", "t3", "b3"),
        issues.Issue(12, "wrong/x", "open", "new", "body", ("coverage", "bug")),
    ]


def test_the_follow_up_pass_files_no_further_bugs():
    row = backend_row(
        "wgsl",
        "output",
        [feature_entry(f"w{i}", "wrong", detail=detail_for(f"w{i}")) for i in range(3)],
    )
    repository = Repository()
    issues.reconcile(
        coverage_report(row), repository, new_bugs=1, log=lambda line: None
    )
    assert sorted(i.key for i in repository.issues) == sorted(
        ["tracking/wgsl/output", bug_for("wgsl", "w0")]
    )


def test_a_dry_run_changes_nothing():
    repository = Repository()
    actions = issues.reconcile(
        coverage_report(hlsl_row()), repository, dry_run=True, log=lambda line: None
    )
    assert len(actions) == 2
    assert repository.issues == []


def test_tracking_issue_closes_at_full_coverage_and_reopens_on_regression():
    state = issues_after(
        issues.plan(coverage_report(hlsl_row(b="supported", c="wrong")), [])
    )
    fixed = coverage_report(hlsl_row(b="supported", c="supported"))
    actions = issues.plan(fixed, state)
    close = [a for a in actions if a.key == "tracking/hlsl/output"]
    assert [(a.kind, a.reason) for a in close] == [("close", "completed")]
    assert close[0].title == "HLSL output coverage: 100.0 % (0 gaps)"
    state = issues_after(actions, state)
    assert issues.plan(fixed, state) == []
    regressed = issues.plan(
        coverage_report(hlsl_row(b="supported", c="refused")), state
    )
    reopen = [a for a in regressed if a.key == "tracking/hlsl/output"]
    assert [a.kind for a in reopen] == ["reopen"]
    assert "1 gap again" in reopen[0].comment
    assert reopen[0].title == "HLSL output coverage: 66.7 % (1 gap)"


def test_bug_issues_close_when_no_probe_fails_their_way():
    state = issues_after(issues.plan(coverage_report(hlsl_row(a="wrong")), []))
    actions = issues.plan(coverage_report(hlsl_row(a="supported", b="refused")), state)
    closes = {a.key: a for a in actions if a.kind == "close"}
    assert set(closes) == {bug_for("hlsl", "a"), bug_for("hlsl", "b")}
    assert {a.reason for a in closes.values()} == {"completed"}
    assert "No HLSL output probe fails this way" in closes[bug_for("hlsl", "a")].comment
    tracking = next(i for i in state if i.key == "tracking/hlsl/output")
    assert f"#{tracking.number}" in closes[bug_for("hlsl", "b")].comment


def test_wrong_results_with_one_cause_are_one_bug_issue():
    detail = 'probe.cu({}): error: identifier "{}" is undefined in device code'
    row = backend_row(
        "cuda",
        "output",
        [
            feature_entry("sinf", "wrong", detail=detail.format(1888, "fin")),
            feature_entry("atomicAdd", "wrong", detail=detail.format(1890, "iin")),
            feature_entry(
                "erff", "wrong", detail='error: identifier "erff" is undefined'
            ),
        ],
    )
    bugs = [
        a for a in issues.plan(coverage_report(row), []) if a.key.startswith("wrong/")
    ]
    assert [a.title for a in bugs] == [
        'CUDA output: identifier "<probe>" is undefined in device code',
        'CUDA output: identifier "erff" is undefined',
    ]
    assert (
        "The coverage probes for 2 CUDA output features give a wrong result" in bugs[
            0
        ].body
    )
    assert "Affected features: `atomicAdd`, `sinf`." in bugs[0].body
    assert "--feature atomicAdd" in bugs[0].body
    assert issues.FEATURES.format("atomicAdd,sinf") in bugs[0].body


def test_a_bug_stays_open_while_one_of_its_features_is_untested():
    row = backend_row(
        "cuda",
        "output",
        [feature_entry(n, "wrong", detail="same failure") for n in ("p", "q")],
    )
    state = issues_after(issues.plan(coverage_report(row), []))
    row["features"] = [feature_entry("p", "supported"), feature_entry("q", "untested")]
    actions = issues.plan(coverage_report(row), state)
    assert [a for a in actions if a.key.startswith("wrong/")] == []


def _wrong(row_features):
    """An HLSL output row of wrong results, from ``(name, detail)`` pairs."""
    return coverage_report(
        backend_row(
            "hlsl",
            "output",
            [feature_entry(n, "wrong", detail=d) for n, d in row_features],
        )
    )


def _open_bugs(state):
    return sorted(
        (i.number, i.title)
        for i in state
        if i.key.startswith("wrong/") and i.state == "open"
    )


def test_a_bug_whose_features_fail_another_way_keeps_its_issue():
    state = issues_after(issues.plan(_wrong([("p", "first failure")]), []))
    (old,) = (i for i in state if i.key.startswith("wrong/"))
    actions = issues.plan(_wrong([("p", "second failure")]), state, new_bugs=0)
    assert [(a.kind, a.number) for a in actions if a.key.startswith("wrong/")] == [
        ("edit", old.number)
    ]
    state = issues_after(actions, state)
    assert _open_bugs(state) == [(old.number, "HLSL output: second failure")]
    assert issues.plan(_wrong([("p", "second failure")]), state) == []


def test_bugs_whose_causes_merge_close_in_favour_of_the_oldest():
    first = _wrong([("p", "'p' is undefined failure"), ("q", "first failure")])
    state = issues_after(issues.plan(first, []))
    oldest, newer = sorted(i.number for i in state if i.key.startswith("wrong/"))
    merged = _wrong([("p", "'p' is undefined"), ("q", "'q' is undefined")])
    actions = issues.plan(merged, state, new_bugs=0)
    close = [a for a in actions if a.kind == "close"]
    assert [(a.number, a.reason) for a in close] == [(newer, "not_planned")]
    assert close[0].comment == (
        f"Its probes still fail, now tracked in #{oldest}, so this issue is closed."
    )
    state = issues_after(actions, state)
    assert _open_bugs(state) == [(oldest, "HLSL output: '<feature>' is undefined")]


def test_a_split_bug_closes_once_each_new_cause_has_an_issue():
    state = issues_after(issues.plan(_wrong([("p", "same"), ("q", "same")]), []))
    (old,) = (i for i in state if i.key.startswith("wrong/"))
    split = _wrong([("p", "one way"), ("q", "another way")])
    capped = issues.plan(split, state, new_bugs=1)
    assert [a.kind for a in capped if a.key.startswith("wrong/")] == ["create"]
    state = issues_after(capped, state)
    filed = issues.plan(split, state)
    assert [a.kind for a in filed if a.key.startswith("wrong/")] == ["create"]
    state = issues_after(filed, state)
    close = [a for a in issues.plan(split, state) if a.kind == "close"]
    assert [a.number for a in close] == [old.number]
    assert close[0].comment == (
        f"Its probes still fail, now tracked in #{old.number + 1}, #{old.number + 2}, so this issue is closed."
    )


def test_a_run_closes_a_split_bug_in_favour_of_the_issues_it_files():
    repository = Repository()
    issues.reconcile(
        _wrong([("p", "same"), ("q", "same")]), repository, log=lambda line: None
    )
    split = _wrong([("p", "one way"), ("q", "another way")])
    actions = issues.reconcile(split, repository, log=lambda line: None)
    assert [a.kind for a in actions if a.key.startswith("wrong/")] == [
        "create",
        "create",
        "close",
    ]
    assert issues.reconcile(split, repository, log=lambda line: None) == []


def test_bug_issues_reopen_when_the_result_is_wrong_again():
    state = issues_after(issues.plan(coverage_report(hlsl_row()), []))
    state = issues_after(
        issues.plan(coverage_report(hlsl_row(b="supported")), state), state
    )
    actions = issues.plan(coverage_report(hlsl_row()), state)
    assert ("reopen", bug_for("hlsl", "b")) in kinds(actions)


def test_untested_and_unmeasured_results_leave_issues_alone():
    state = issues_after(issues.plan(coverage_report(hlsl_row()), []))
    assert [
        a
        for a in issues.plan(coverage_report(hlsl_row(b="untested")), state)
        if "wrong/" in a.key
    ] == []
    # A row that is missing, measured nothing, or is not offered says nothing about its issues.
    unmeasured = backend_row("hlsl", "output", [feature_entry("b", "untested")])
    not_offered = {
        "language": "hlsl",
        "direction": "output",
        "offered": False,
        "reason": "r",
    }
    for report in (
        coverage_report(),
        coverage_report(unmeasured),
        coverage_report(not_offered),
    ):
        assert issues.plan(report, state) == []


def test_bug_for_a_feature_dropped_from_the_list_is_closed_as_not_planned():
    state = issues_after(issues.plan(coverage_report(hlsl_row()), []))
    row = hlsl_row()
    row["features"] = [f for f in row["features"] if f["name"] != "b"]
    actions = issues.plan(coverage_report(row), state)
    assert [(a.kind, a.reason) for a in actions if a.key.startswith("wrong/")] == [
        ("close", "not_planned")
    ]


def test_new_bug_issues_are_capped_and_spread_over_rows():
    many = backend_row(
        "wgsl",
        "output",
        [
            feature_entry(f"w{i:02}", "wrong", detail=detail_for(f"w{i:02}"))
            for i in range(20)
        ],
    )
    few = backend_row(
        "glsl",
        "output",
        [feature_entry(f"g{i}", "wrong", detail=detail_for(f"g{i}")) for i in range(2)],
    )
    actions = issues.plan(coverage_report(many, few), [], new_bugs=4)
    bugs = [a.key for a in actions if a.key.startswith("wrong/")]
    assert bugs == [
        bug_for("wgsl", "w00"),
        bug_for("glsl", "g0"),
        bug_for("wgsl", "w01"),
        bug_for("glsl", "g1"),
    ]
    state = issues_after(actions)
    following = [
        a.key
        for a in issues.plan(coverage_report(many, few), state, new_bugs=4)
        if a.kind == "create"
    ]
    assert following == [bug_for("wgsl", f"w{i:02}") for i in range(2, 6)]


def test_existing_bug_issues_are_updated_even_past_the_cap():
    state = issues_after(issues.plan(coverage_report(hlsl_row()), []))
    state = issues_after(
        issues.plan(coverage_report(hlsl_row(b="supported")), state), state
    )
    actions = issues.plan(coverage_report(hlsl_row()), state, new_bugs=0)
    assert ("reopen", bug_for("hlsl", "b")) in kinds(actions)


def test_edits_add_missing_labels_and_keep_the_others():
    state = issues_after(issues.plan(coverage_report(hlsl_row()), []))
    trimmed = [replace(i, labels=("coverage", "good first issue")) for i in state]
    actions = issues.plan(coverage_report(hlsl_row()), trimmed)
    tracking = next(a for a in actions if a.key == "tracking/hlsl/output")
    assert tracking.kind == "edit"
    assert tracking.labels == ("backend:hlsl", "direction:output", "area:codegen")


def test_issues_without_a_marker_or_with_a_duplicate_key_are_not_managed():
    report = coverage_report(hlsl_row())
    repository = Repository()
    issues.reconcile(report, repository, log=lambda line: None)
    state = repository.issues
    foreign = issues.Issue(
        99, "", "open", "HLSL output coverage", "written by a person"
    )
    copy = replace(
        next(i for i in state if i.key == "tracking/hlsl/output"),
        number=100,
        body="copied",
    )
    assert issues.plan(report, [*state, foreign, copy], "o/r") == []


def test_crossgl_ir_row_gets_a_tracking_issue_but_no_bug_issues():
    row = backend_row(
        "crossgl",
        "ir",
        [
            feature_entry("OpX", "wrong", detail="wgsl: no"),
            feature_entry("OpY", "supported"),
        ],
    )
    actions = issues.plan(coverage_report(row), [])
    assert kinds(actions) == [("create", "tracking/crossgl/ir")]
    assert actions[0].title == "CrossGL IR expressiveness: 50.0 % (1 gap)"
    assert actions[0].labels == ("coverage", "backend:crossgl", "area:frontend")
    assert "#crossgl-ir-expressiveness" in actions[0].body


def test_input_rows_describe_the_round_trip():
    row = backend_row("glsl", "input", [feature_entry("x", "wrong", detail="bad")])
    actions = issues.plan(coverage_report(row), [])
    assert actions[0].labels[-1] == "area:frontend"
    assert "translates it back" in actions[0].body
    assert "round trip" in actions[1].body


def test_spirv_input_bugs_name_the_dropped_instruction_as_a_failure():
    detail = "the round trip no longer calls NMin"
    row = backend_row("spirv", "input", [feature_entry("NMin", "wrong", detail=detail)])
    body = issues.plan(coverage_report(row), [])[1].body
    assert "or that no longer calls the feature's instruction" in body
    assert detail in body
    glsl = backend_row("glsl", "input", [feature_entry("x", "wrong", detail="bad")])
    assert "instruction" not in issues.plan(coverage_report(glsl), [])[1].body


def test_spirv_output_bugs_name_ignored_nan_operands_as_a_failure():
    detail = "the translation calls no NMin and tests no operand for NaN"
    row = backend_row(
        "spirv", "output", [feature_entry("NMin", "wrong", detail=detail)]
    )
    body = issues.plan(coverage_report(row), [])[1].body
    assert "or that ignores the NaN operands the feature's instruction handles" in body
    assert detail in body
    glsl = backend_row("glsl", "output", [feature_entry("x", "wrong", detail="bad")])
    assert "NaN" not in issues.plan(coverage_report(glsl), [])[1].body


def test_long_gap_lists_are_cut_to_the_body_limit(monkeypatch):
    monkeypatch.setattr(issues, "BODY_LIMIT", 1_500)
    row = backend_row(
        "wgsl",
        "output",
        [feature_entry(f"w{i:03}", "refused", detail="x" * 100) for i in range(50)],
    )
    body = issues.plan(coverage_report(row), [])[0].body
    assert len(body) <= 1_600
    assert body.rstrip().endswith("more gaps are listed in COVERAGE.md.")
