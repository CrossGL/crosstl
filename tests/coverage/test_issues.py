import json
from dataclasses import replace

import pytest

from tools.coverage import __main__ as cli
from tools.coverage import issues


def _feature(name, outcome, category="function", detail=""):
    feature = {"name": name, "category": category, "group": "g", "outcome": outcome}
    return feature | ({"detail": detail} if detail else {})


def _row(language, direction, features):
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


def _report(*rows):
    return {
        "commit": "0123456789abcdef",
        "date": "2026-10-19T03:41:00Z",
        "backends": list(rows),
    }


def _hlsl(**outcomes):
    """An HLSL output row; ``outcomes`` overrides the outcome of the named features."""
    base = {"a": "supported", "b": "wrong", "c": "refused", "d": "untested"}
    return _row(
        "hlsl",
        "output",
        [
            _feature(name, outcome, detail=f"{name} failed")
            for name, outcome in (base | outcomes).items()
        ],
    )


def _bug(language, name, direction="output"):
    """The bug key of feature ``name`` in the rows built here, whose detail is ``"<name> failed"``."""
    row = {"language": language, "direction": direction}
    return issues.bug_key(row, _feature(name, "wrong", detail=f"{name} failed"))


def _applied(actions, existing=()):
    """The managed issues after ``actions`` run, numbering new issues after the existing ones."""
    state = {issue.key: issue for issue in existing}
    number = max((issue.number for issue in existing), default=0)
    for action in actions:
        if action.kind == "create":
            number += 1
            state[action.key] = issues.Issue(
                number, action.key, "open", action.title, action.body, action.labels
            )
            continue
        old = state[action.key]
        status = {"close": "closed", "reopen": "open"}.get(action.kind, old.state)
        labels = old.labels + tuple(
            label for label in action.labels if label not in old.labels
        )
        state[action.key] = replace(
            old, state=status, title=action.title, body=action.body, labels=labels
        )
    return list(state.values())


def _kinds(actions):
    return sorted((action.kind, action.key) for action in actions)


def test_keys_are_marker_safe_and_round_trip():
    key = issues.key("wrong", "hlsl", "output", "Texture2D<float4>::Sample x")
    assert key == "wrong/hlsl/output/Texture2D_float4_::Sample_x"
    assert issues.marker_key("text\n" + issues.MARKER.format(key) + "\nmore") == key
    assert issues.marker_key("an ordinary issue") == ""
    assert issues.marker_key(None) == ""


def test_first_run_files_tracking_and_bug_issues():
    report = _report(_hlsl(), _row("glsl", "output", [_feature("x", "supported")]))
    actions = issues.plan(report, [])
    assert _kinds(actions) == [
        ("create", "tracking/hlsl/output"),
        ("create", _bug("hlsl", "b")),
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
    assert "- [ ] `b` wrong: b failed" in tracking.body
    assert "- [ ] `c` refused: c failed" in tracking.body
    assert "`d`" not in tracking.body
    bug = actions[1]
    assert bug.title == "HLSL output: b failed"
    assert (
        "The coverage probe for `b` (HLSL output, g) gives a wrong result." in bug.body
    )
    assert bug.labels[0] == "bug"
    assert (
        "python -m tools.coverage run --backend hlsl --direction output --feature b"
        in bug.body
    )


class _Repository(issues.GitHub):
    """An in-memory repository of managed issues."""

    def __init__(self, existing=()):
        super().__init__("o/r")
        self.issues = list(existing)

    def managed(self):
        return list(self.issues)

    def apply(self, action):
        self.issues = _applied([action], self.issues)
        return max(i.number for i in self.issues)


def test_a_run_links_the_bugs_it_files_and_the_next_run_changes_nothing():
    report = _report(_hlsl())
    repository = _Repository()
    logged = []
    actions = issues.reconcile(report, repository, log=logged.append)
    assert _kinds(actions) == [
        ("create", "tracking/hlsl/output"),
        ("create", _bug("hlsl", "b")),
        ("edit", "tracking/hlsl/output"),
    ]
    assert len(logged) == 3
    bug = next(i for i in repository.issues if i.key == _bug("hlsl", "b"))
    tracking = next(i for i in repository.issues if i.key == "tracking/hlsl/output")
    assert f"- [ ] `b` wrong (#{bug.number}): b failed" in tracking.body
    assert issues.reconcile(report, repository, log=logged.append) == []
    assert len(logged) == 3


def test_the_follow_up_pass_files_no_further_bugs():
    row = _row(
        "wgsl",
        "output",
        [_feature(f"w{i}", "wrong", detail=f"w{i} failed") for i in range(3)],
    )
    repository = _Repository()
    issues.reconcile(_report(row), repository, new_bugs=1, log=lambda line: None)
    assert sorted(i.key for i in repository.issues) == sorted(
        ["tracking/wgsl/output", _bug("wgsl", "w0")]
    )


def test_a_dry_run_changes_nothing():
    repository = _Repository()
    actions = issues.reconcile(
        _report(_hlsl()), repository, dry_run=True, log=lambda line: None
    )
    assert len(actions) == 2
    assert repository.issues == []


def test_tracking_issue_closes_at_full_coverage_and_reopens_on_regression():
    state = _applied(issues.plan(_report(_hlsl(b="supported", c="wrong")), []))
    fixed = _report(_hlsl(b="supported", c="supported"))
    actions = issues.plan(fixed, state)
    close = [a for a in actions if a.key == "tracking/hlsl/output"]
    assert [(a.kind, a.reason) for a in close] == [("close", "completed")]
    assert close[0].title == "HLSL output coverage: 100.0 % (0 gaps)"
    state = _applied(actions, state)
    assert issues.plan(fixed, state) == []
    regressed = issues.plan(_report(_hlsl(b="supported", c="refused")), state)
    reopen = [a for a in regressed if a.key == "tracking/hlsl/output"]
    assert [a.kind for a in reopen] == ["reopen"]
    assert "1 gap again" in reopen[0].comment
    assert reopen[0].title == "HLSL output coverage: 66.7 % (1 gap)"


def test_bug_issues_close_when_no_probe_fails_their_way():
    state = _applied(issues.plan(_report(_hlsl(a="wrong")), []))
    actions = issues.plan(_report(_hlsl(a="supported", b="refused")), state)
    closes = {a.key: a for a in actions if a.kind == "close"}
    assert set(closes) == {_bug("hlsl", "a"), _bug("hlsl", "b")}
    assert {a.reason for a in closes.values()} == {"completed"}
    assert "No HLSL output probe fails this way" in closes[_bug("hlsl", "a")].comment
    tracking = next(i for i in state if i.key == "tracking/hlsl/output")
    assert f"#{tracking.number}" in closes[_bug("hlsl", "b")].comment


def test_wrong_results_with_one_cause_are_one_bug_issue():
    detail = 'probe.cu({}): error: identifier "{}" is undefined in device code'
    row = _row(
        "cuda",
        "output",
        [
            _feature("sinf", "wrong", detail=detail.format(1888, "fin")),
            _feature("atomicAdd", "wrong", detail=detail.format(1890, "iin")),
            _feature("erff", "wrong", detail='error: identifier "erff" is undefined'),
        ],
    )
    bugs = [a for a in issues.plan(_report(row), []) if a.key.startswith("wrong/")]
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
    row = _row(
        "cuda",
        "output",
        [_feature(n, "wrong", detail="same failure") for n in ("p", "q")],
    )
    state = _applied(issues.plan(_report(row), []))
    row["features"] = [_feature("p", "supported"), _feature("q", "untested")]
    actions = issues.plan(_report(row), state)
    assert [a for a in actions if a.key.startswith("wrong/")] == []


def test_causes_keep_the_digits_of_names():
    first = issues.cause(
        {"detail": "probe.hlsl:3:9: error: undeclared identifier 'packSnorm2x16'"}
    )
    second = issues.cause(
        {"detail": "probe.hlsl:4:9: error: undeclared identifier 'packSnorm4x8'"}
    )
    assert first == "probe.hlsl:N:N: error: undeclared identifier 'packSnorm2x16'"
    assert first != second


def test_bug_issues_reopen_when_the_result_is_wrong_again():
    state = _applied(issues.plan(_report(_hlsl()), []))
    state = _applied(issues.plan(_report(_hlsl(b="supported")), state), state)
    actions = issues.plan(_report(_hlsl()), state)
    assert ("reopen", _bug("hlsl", "b")) in _kinds(actions)


def test_untested_and_unmeasured_results_leave_issues_alone():
    state = _applied(issues.plan(_report(_hlsl()), []))
    assert [
        a for a in issues.plan(_report(_hlsl(b="untested")), state) if "wrong/" in a.key
    ] == []
    # A row that is missing, measured nothing, or is not offered says nothing about its issues.
    unmeasured = _row("hlsl", "output", [_feature("b", "untested")])
    not_offered = {
        "language": "hlsl",
        "direction": "output",
        "offered": False,
        "reason": "r",
    }
    for report in (_report(), _report(unmeasured), _report(not_offered)):
        assert issues.plan(report, state) == []


def test_bug_for_a_feature_dropped_from_the_list_is_closed_as_not_planned():
    state = _applied(issues.plan(_report(_hlsl()), []))
    row = _hlsl()
    row["features"] = [f for f in row["features"] if f["name"] != "b"]
    actions = issues.plan(_report(row), state)
    assert [(a.kind, a.reason) for a in actions if a.key.startswith("wrong/")] == [
        ("close", "not_planned")
    ]


def test_new_bug_issues_are_capped_and_spread_over_rows():
    many = _row(
        "wgsl",
        "output",
        [_feature(f"w{i:02}", "wrong", detail=f"w{i:02} failed") for i in range(20)],
    )
    few = _row(
        "glsl",
        "output",
        [_feature(f"g{i}", "wrong", detail=f"g{i} failed") for i in range(2)],
    )
    actions = issues.plan(_report(many, few), [], new_bugs=4)
    bugs = [a.key for a in actions if a.key.startswith("wrong/")]
    assert bugs == [
        _bug("wgsl", "w00"),
        _bug("glsl", "g0"),
        _bug("wgsl", "w01"),
        _bug("glsl", "g1"),
    ]
    state = _applied(actions)
    following = [
        a.key
        for a in issues.plan(_report(many, few), state, new_bugs=4)
        if a.kind == "create"
    ]
    assert following == [_bug("wgsl", f"w{i:02}") for i in range(2, 6)]


def test_existing_bug_issues_are_updated_even_past_the_cap():
    state = _applied(issues.plan(_report(_hlsl()), []))
    state = _applied(issues.plan(_report(_hlsl(b="supported")), state), state)
    actions = issues.plan(_report(_hlsl()), state, new_bugs=0)
    assert ("reopen", _bug("hlsl", "b")) in _kinds(actions)


def test_edits_add_missing_labels_and_keep_the_others():
    state = _applied(issues.plan(_report(_hlsl()), []))
    trimmed = [replace(i, labels=("coverage", "good first issue")) for i in state]
    actions = issues.plan(_report(_hlsl()), trimmed)
    tracking = next(a for a in actions if a.key == "tracking/hlsl/output")
    assert tracking.kind == "edit"
    assert tracking.labels == ("backend:hlsl", "direction:output", "area:codegen")


def test_issues_without_a_marker_or_with_a_duplicate_key_are_not_managed():
    report = _report(_hlsl())
    repository = _Repository()
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
    row = _row(
        "crossgl",
        "ir",
        [_feature("OpX", "wrong", detail="wgsl: no"), _feature("OpY", "supported")],
    )
    actions = issues.plan(_report(row), [])
    assert _kinds(actions) == [("create", "tracking/crossgl/ir")]
    assert actions[0].title == "CrossGL IR expressiveness: 50.0 % (1 gap)"
    assert actions[0].labels == ("coverage", "backend:crossgl", "area:frontend")
    assert "#crossgl-ir-expressiveness" in actions[0].body


def test_input_rows_describe_the_round_trip():
    row = _row("glsl", "input", [_feature("x", "wrong", detail="bad")])
    actions = issues.plan(_report(row), [])
    assert actions[0].labels[-1] == "area:frontend"
    assert "translates it back" in actions[0].body
    assert "round trip" in actions[1].body


def test_long_gap_lists_are_cut_to_the_body_limit(monkeypatch):
    monkeypatch.setattr(issues, "BODY_LIMIT", 1_500)
    row = _row(
        "wgsl",
        "output",
        [_feature(f"w{i:03}", "refused", detail="x" * 100) for i in range(50)],
    )
    body = issues.plan(_report(row), [])[0].body
    assert len(body) <= 1_600
    assert body.rstrip().endswith("more gaps are listed in COVERAGE.md.")


class _FakeGitHub(issues.GitHub):
    def __init__(self, pages):
        super().__init__("o/r")
        self.pages = pages
        self.calls = []

    def _api(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET":
            return self.pages.pop(0)
        return {"number": 7}


def _item(number, body, **extra):
    return {
        "number": number,
        "state": "open",
        "title": "t",
        "body": body,
        "labels": [{"name": "coverage"}],
    } | extra


def test_github_lists_only_marked_issues_across_pages():
    marked = issues.MARKER.format("tracking/hlsl/output")
    first = [_item(n, "no marker") for n in range(1, 100)] + [_item(100, marked)]
    second = [_item(101, marked, pull_request={}), _item(102, None)]
    github = _FakeGitHub([first, second])
    found = github.managed()
    assert [(i.number, i.key, i.labels) for i in found] == [
        (100, "tracking/hlsl/output", ("coverage",))
    ]
    assert [call[1] for call in github.calls] == [
        "issues?state=all&labels=coverage&per_page=100&page=1",
        "issues?state=all&labels=coverage&per_page=100&page=2",
    ]


def test_github_comments_before_closing_and_adds_labels_last():
    github = _FakeGitHub([])
    action = issues.Action("close", "k", "t", "b", ("bug",), 5, "why", "completed")
    assert github.apply(action) == 5
    assert github.calls == [
        ("POST", "issues/5/comments", {"body": "why"}),
        (
            "PATCH",
            "issues/5",
            {"title": "t", "body": "b", "state": "closed", "state_reason": "completed"},
        ),
        ("POST", "issues/5/labels", {"labels": ["bug"]}),
    ]
    assert github.apply(issues.Action("create", "k", "t", "b", ("bug",))) == 7
    assert github.calls[-1] == (
        "POST",
        "issues",
        {"title": "t", "body": "b", "labels": ["bug"]},
    )


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_prints_the_plan_and_applies_it_unless_dry_run(
    tmp_path, monkeypatch, capsys, dry_run
):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_report(_hlsl())))
    applied = []
    monkeypatch.setattr(issues.GitHub, "managed", lambda self: [])
    monkeypatch.setattr(
        issues.GitHub, "apply", lambda self, action: applied.append(action.key)
    )
    argv = ["issues", str(path), "--repository", "o/r"] + (
        ["--dry-run"] if dry_run else []
    )
    assert cli.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("create new issue  HLSL output coverage: 33.3 % (2 gaps)")
    # The fake repository never records the bug, so the applied run plans its follow-up pass on an empty list.
    expected = 2 if dry_run else 3
    assert out[-1] == f"{expected} issue changes" + (" planned" if dry_run else "")
    assert applied == (
        []
        if dry_run
        else ["tracking/hlsl/output", _bug("hlsl", "b"), "tracking/hlsl/output"]
    )


def test_cli_reports_github_errors(tmp_path, monkeypatch, capsys):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_report()))

    def fail(self):
        raise issues.GitHubError("gh api GET issues: HTTP 401")

    monkeypatch.setattr(issues.GitHub, "managed", fail)
    assert cli.main(["issues", str(path)]) == 2
    assert "HTTP 401" in capsys.readouterr().err
