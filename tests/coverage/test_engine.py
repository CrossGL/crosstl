import hashlib
import json

import pytest

from tools.coverage import report, runner, upstream, validators
from tools.coverage.model import Feature, Result, Score, Signature, Type

ABS = Feature(
    "glsl", "function", "abs", Signature(Type("float", (3,)), ()), crossgl="abs"
)


def test_score_leaves_untested_out_of_the_denominator():
    score = Score()
    for outcome in (
        "supported",
        "supported",
        "supported",
        "refused",
        "untested",
        "untested",
    ):
        score.add(outcome)
    assert score.measured == 4
    assert score.percent == 75.0
    assert Score().percent is None


def test_report_groups_results_and_marks_directions_not_offered():
    results = [
        Result(ABS, "output", "supported", validator="glslangValidator"),
        Result(
            Feature("glsl", "function", "texture", None),
            "output",
            "untested",
            "no probe",
        ),
    ]
    data = report.build(results, ["glsl", "wgsl"], ["output", "input"])
    backends = {(b["language"], b["direction"]): b for b in data["backends"]}
    assert backends["glsl", "output"]["score"] == 100.0
    assert backends["glsl", "output"]["counts"]["untested"] == 1
    assert [f["name"] for f in backends["glsl", "output"]["features"]] == [
        "abs",
        "texture",
    ]
    assert backends["wgsl", "input"]["offered"] is False
    assert ("glsl", "input") not in backends
    assert backends["glsl", "output"]["validator"] == "glslangValidator"
    assert "not offered" in report.summary(data)


def test_coverage_json_stores_features_as_rows_and_each_diagnostic_once(tmp_path):
    texture = Feature("glsl", "function", "texture", None)
    results = [
        Result(ABS, "output", "supported"),
        Result(texture, "output", "wrong", "ERROR: 'texture' : no matching overload"),
        Result(
            Feature("glsl", "function", "textureLod", None),
            "output",
            "wrong",
            "ERROR: 'texture' : no matching overload",
        ),
        Result(
            Feature("glsl", "function", "fma", None),
            "output",
            "refused",
            "no lowering",
            semantic="GLSL.std.450:Fma",
        ),
    ]
    data = report.build(results, ["glsl", "wgsl"], ["output", "input"])
    path = tmp_path / "coverage.json"
    report.write(data, path)

    stored = {b["language"]: b for b in json.loads(path.read_text())["backends"]}
    glsl = stored["glsl"]
    assert glsl["columns"][:5] == ["name", "category", "group", "outcome", "detail"]
    assert glsl["details"] == ["ERROR: 'texture' : no matching overload", "no lowering"]
    assert [row[4] for row in glsl["features"]] == [None, 1, 0, 0]
    assert "\n" not in path.read_text().rstrip("\n")
    assert report.load(path) == data
    # A backend that is not offered has no features to store as rows.
    assert stored["wgsl"] == next(
        b for b in data["backends"] if b["language"] == "wgsl"
    )


def test_loading_keeps_features_that_are_already_dictionaries():
    data = {
        "schema": 1,
        "backends": [
            {"language": "glsl", "features": [{"name": "abs", "outcome": "supported"}]}
        ],
    }
    assert report.loads(json.dumps(data)) == data


class Diagnostic(ValueError):
    project_diagnostic_code = "project.translate.example"


@pytest.mark.parametrize(
    ("error", "expected"),
    [(Diagnostic("no lowering"), runner.Refused), (AttributeError("x"), runner.Broken)],
)
def test_only_coded_diagnostics_count_as_refusals(monkeypatch, error, expected):
    import crosstl

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(crosstl, "translate", fail)
    with pytest.raises(expected):
        runner.translate("shader S {}", ".cgl", "opengl", "cgl")


@pytest.mark.skipif(
    not validators.available("glsl"), reason="glslangValidator is not installed"
)
def test_output_probe_end_to_end():
    from tools.coverage.model import Param

    vec3 = Type("float", (3,))
    feature = Feature(
        "glsl", "function", "abs", Signature(vec3, (Param(vec3),)), crossgl="abs"
    )
    result = runner.run_output(feature, "glsl")
    assert (result.outcome, result.validator) == (
        "supported",
        "glslangValidator",
    ), result.detail


# The shape of slangc 2026.9's log for an undefined name: the header names the kind of error, the label the name.
SLANGC_LOG = """error[E30015]: undefined identifier
  --> probe.slang:10:15
   |
10 | uint2 r = {name}(a0);
   |           ^^^^^^^^ undefined identifier '{name}'.
---'
error[E30019]: type mismatch in expression
  --> probe.slang:11:5
   |
11 | outp[i] = r;
   |     ^ expected an expression of type 'float', got 'uint2'
---'"""


@pytest.mark.parametrize(
    ("language", "log", "expected"),
    [
        (
            "slang",
            SLANGC_LOG.format(name="bitCount"),
            "error[E30015]: undefined identifier 'bitCount'",
        ),
        (
            "slang",
            "error[E30011]: left of '=' is not an l-value\n --> probe.slang:9:5\n"
            "  |\n9 | f(a0) = r;\n  | ^ left of '=' is not an l-value.",
            "error[E30011]: left of '=' is not an l-value",
        ),
        (
            "slang",
            SLANGC_LOG.split("---'\n", 1)[1],
            "error[E30019]: type mismatch in expression: expected an expression of type"
            " 'float', got 'uint2'",
        ),
        (
            "slang",
            "error[E30015]: undefined identifier",
            "error[E30015]: undefined identifier",
        ),
        # Other tools' first error line already carries the particulars.
        ("rust", SLANGC_LOG.format(name="x"), "error[E30015]: undefined identifier"),
        (
            "glsl",
            "note: x\nERROR: 0:4: 'f' : no matching function",
            "ERROR: 0:4: 'f' : no matching function",
        ),
        ("glsl", "nothing to report", "nothing to report"),
    ],
)
def test_first_error_joins_slangc_headers_with_their_label(language, log, expected):
    assert validators.first_error(log, language) == expected


def test_slang_probes_failing_on_different_names_have_different_causes(monkeypatch):
    from tools.coverage import causes

    monkeypatch.setattr(
        validators,
        "validate",
        lambda language, source, *args: validators.Verdict(
            False, SLANGC_LOG.format(name=source)
        ),
    )
    names = ("bitCount", "findMSB", "gl_NumWorkGroups")
    details = [runner._validated("slang", name, "compute").log for name in names]
    features = [
        {"name": feature, "crossgl": crossgl, "detail": detail}
        for (feature, crossgl), detail in zip(
            [
                ("countbits", "bitCount"),
                ("firstbithigh", "findMSB"),
                ("num-workgroups", None),
            ],
            details,
        )
    ]
    found = [causes.cause(feature, "slang") for feature in features]
    assert found[0] == found[1] == "error[E30015]: undefined identifier '<feature>'"
    assert found[2] == "error[E30015]: undefined identifier 'gl_NumWorkGroups'"


def test_a_run_validates_each_distinct_source_once(monkeypatch):
    calls = []

    def validate(language, source, stage="compute", entry="", target=""):
        calls.append(source)
        return validators.Verdict("bad" not in source, "error: bad probe")

    monkeypatch.setattr(validators, "validate", validate)
    monkeypatch.setattr(
        runner, "_job", lambda job: runner._validated("glsl", job, "compute")
    )
    verdicts = runner.run(["good", "bad", "good", "bad"], workers=1)
    assert [v.accepted for v in verdicts] == [True, False, True, False]
    assert verdicts[3].log == "error: bad probe"
    assert calls == ["good", "bad"]
    # Nothing carries over into the next run, which may use another tool.
    runner.run(["good"], workers=1)
    assert calls == ["good", "bad", "good"]
    assert runner._verdicts is None


def test_fetch_rejects_a_changed_upstream_file(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSSTL_COVERAGE_CACHE", str(tmp_path))
    source = upstream.SOURCES["glsl"]
    cached = tmp_path / "upstream" / source.commit / source.path
    cached.parent.mkdir(parents=True)
    cached.write_text("tampered")
    with pytest.raises(upstream.UpstreamError, match="sha256"):
        upstream.fetch("glsl", offline=True)
    cached.write_bytes(b"pinned")
    monkeypatch.setitem(
        upstream.SOURCES,
        "glsl",
        upstream.Source(
            source.repository,
            source.commit,
            source.path,
            hashlib.sha256(b"pinned").hexdigest(),
            "",
        ),
    )
    assert upstream.fetch("glsl", offline=True) == "pinned"


def _row(language, direction, outcome, **extra):
    return {
        "language": language,
        "direction": direction,
        "offered": True,
        "score": 100.0 if outcome == "supported" else 0.0,
        "counts": {},
        "features": [
            {
                "name": "fma",
                "category": "function",
                "group": "g",
                "outcome": outcome,
                **extra,
            }
        ],
    }


def _report(*rows, commit="abc", tool="glslangValidator"):
    return {
        "schema": 1,
        "commit": commit,
        "date": "2026-10-19T00:00:00Z",
        "crosstl": "",
        "sources": {},
        "validators": {tool: "1.0"},
        "backends": list(rows),
    }


def test_merge_combines_runners_and_rebuilds_the_ir_row():
    linux = _report(_row("glsl", "output", "wrong", semantic="ext:Fma"))
    mac = _report(_row("metal", "output", "supported"), tool="metal")
    merged = report.merge([linux, mac])
    assert [(b["language"], b["direction"]) for b in merged["backends"]] == [
        ("glsl", "output"),
        ("metal", "output"),
        ("crossgl", "ir"),
    ]
    assert merged["validators"] == {"glslangValidator": "1.0", "metal": "1.0"}
    # Metal borrows GLSL's semantic id, so the two rows are one IR feature, expressible through Metal.
    (feature,) = merged["backends"][-1]["features"]
    assert feature["outcome"] == "supported" and feature["via"] == ["metal"]


def test_merge_refuses_mixed_commits_and_double_measurements():
    row = _row("glsl", "output", "supported")
    with pytest.raises(ValueError, match="different commits"):
        report.merge([_report(row), _report(row, commit="def")])
    with pytest.raises(ValueError, match="more than one report"):
        report.merge([_report(row), _report(row)])


def test_run_measures_only_the_requested_features(monkeypatch, tmp_path, capsys):
    from tools.coverage import __main__ as cli
    from tools.coverage import features

    other = Feature("glsl", "function", "sin", ABS.signature, crossgl="sin")
    monkeypatch.setattr(features, "load", lambda language, offline=False: [ABS, other])
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(validators, "can_lower", lambda language, target="": True)
    measured = []

    def run(jobs, workers=None):
        measured.extend((f.name, direction) for f, direction, _ in jobs)
        return [Result(f, direction, "supported") for f, direction, _ in jobs]

    monkeypatch.setattr(runner, "run", run)
    argv = [
        "run",
        "--backend",
        "glsl",
        "--feature",
        "abs",
        "--output",
        str(tmp_path / "c.json"),
    ]
    assert cli.main(argv) == 0
    assert measured == [("abs", "output"), ("abs", "input")]
    capsys.readouterr()


def test_tools_lists_every_language_and_how_to_provide_a_missing_tool(
    monkeypatch, capsys
):
    from tools.coverage import __main__ as cli

    monkeypatch.setattr(
        validators, "available", lambda language, target="": language == "glsl"
    )
    monkeypatch.setattr(validators, "can_lower", lambda language: language == "glsl")
    monkeypatch.setattr(validators, "version", lambda tool: "Glslang Version: 11")
    assert cli.main(["tools"]) == 0
    lines = capsys.readouterr().out.splitlines()
    rows = {line.split()[0]: line for line in lines[1 : len(cli.LANGUAGES) + 1]}
    assert sorted(rows) == cli.LANGUAGES
    assert rows["glsl"].endswith("Glslang Version: 11")
    assert "output, input" in rows["glsl"]
    assert rows["wgsl"].endswith("no: put naga on PATH or set CROSSTL_COVERAGE_NAGA")
    assert "CROSSTL_COVERAGE_RUST_GPU" in rows["rust"]
    assert "MetalToolchain" in rows["metal"]
    assert "glsl yes, hlsl no" in lines[-2]


def test_strict_run_names_exactly_the_missing_lowering_tool(monkeypatch, capsys):
    from tools.coverage import __main__ as cli

    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(validators, "can_lower", lambda language: False)
    monkeypatch.setattr(
        validators, "command", lambda tool: () if tool == "spirv-dis" else (tool,)
    )
    assert cli.main(["run", "--strict", "--backend", "rust"]) == 2
    assert capsys.readouterr().err.strip() == (
        "error: missing tools for rust lowering to SPIR-V (spirv-dis)"
    )


@pytest.mark.parametrize(
    ("banner", "found"),
    [("Mojo 1.1.0 (8189361e)", True), ("mojo 24.4.0 (59977802)", False)],
)
def test_mojo_counts_as_installed_only_for_the_pinned_release_line(
    tmp_path, monkeypatch, banner, found
):
    fake = tmp_path / "mojo"
    fake.write_text(f"#!/bin/sh\necho '{banner}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("CROSSTL_COVERAGE_MOJO", str(fake))
    validators.command.cache_clear()
    try:
        assert validators.command("mojo") == ((str(fake),) if found else ())
    finally:
        validators.command.cache_clear()
