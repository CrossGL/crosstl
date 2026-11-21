import hashlib
import json
import os

import pytest

from tools.coverage import lowering, publish, report, runner, upstream, validators
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


def _split(tmp_path, data):
    for name, text in report.site(data).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path / "coverage.json"


def test_the_coverage_branch_keeps_each_backends_features_in_its_own_file(tmp_path):
    data = report.build(
        [Result(ABS, "output", "supported"), Result(ABS, "input", "wrong", "bad")],
        ["glsl", "wgsl"],
        ["output", "input"],
    )
    path = _split(tmp_path, data)

    index = {
        (b["language"], b["direction"]): b
        for b in json.loads(path.read_text())["backends"]
    }
    assert index["glsl", "input"]["file"] == "features/glsl-input.json"
    assert index["glsl", "input"]["counts"]["wrong"] == 1
    assert not any("features" in b for b in index.values())
    # A backend that is not offered has no feature file.
    assert "file" not in index["wgsl", "input"]
    stored = json.loads((tmp_path / "features/glsl-input.json").read_text())
    assert (stored["language"], stored["direction"], stored["details"]) == (
        "glsl",
        "input",
        ["bad"],
    )
    assert report.load(path) == data


@pytest.mark.parametrize(
    "name, error",
    [
        ("../features/glsl-output.json", "not the name of a feature file"),
        ("/etc/passwd", "not the name of a feature file"),
        ("features/glsl-input.json", "holds the features of glsl input"),
    ],
)
def test_loading_reads_only_the_feature_file_of_the_same_backend(tmp_path, name, error):
    data = report.build(
        [Result(ABS, "output", "supported"), Result(ABS, "input", "supported")],
        ["glsl"],
        ["output", "input"],
    )
    path = _split(tmp_path, data)
    index = json.loads(path.read_text())
    next(b for b in index["backends"] if b["direction"] == "output")["file"] = name
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match=error):
        report.load(path)


def test_loading_refuses_a_feature_file_linked_outside_the_report(tmp_path):
    data = report.build([Result(ABS, "output", "supported")], ["glsl"], ["output"])
    path = _split(tmp_path / "site", data)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (path.parent / "features/glsl-output.json").replace(outside / "glsl-output.json")
    (path.parent / "features/glsl-output.json").symlink_to(outside / "glsl-output.json")
    with pytest.raises(ValueError, match="leads outside"):
        report.load(path)


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


def _counting_validator(monkeypatch, calls, versions):
    def validate(language, source, stage="compute", entry="", target=""):
        calls.append(source)
        if source == "slow":
            return validators.Verdict(False, "dxc timed out after 60 s", timed_out=True)
        return validators.Verdict("bad" not in source, "error: bad probe")

    monkeypatch.setattr(validators, "validate", validate)
    monkeypatch.setattr(validators, "version", lambda tool: versions[tool])
    monkeypatch.setattr(
        runner, "_job", lambda job: runner._validated("hlsl", job, "compute")
    )
    runner._checker.cache_clear()


def test_kept_verdicts_carry_over_for_the_same_tool(monkeypatch, tmp_path):
    calls, versions = [], {"dxc": "dxcompiler.dll: 1.8"}
    _counting_validator(monkeypatch, calls, versions)
    monkeypatch.setenv(runner.VERDICTS, str(tmp_path))
    first = runner.run(["good", "bad", "slow"], workers=1)
    second = runner.run(["good", "bad", "slow"], workers=1)
    assert first == second
    assert second[1] == validators.Verdict(False, "error: bad probe")
    # A timeout depends on the machine's load, so it is checked again.
    assert calls == ["good", "bad", "slow", "slow"]
    # Another tool version checks everything again.
    versions["dxc"] = "dxcompiler.dll: 1.9"
    runner._checker.cache_clear()
    runner.run(["good"], workers=1)
    assert calls[4:] == ["good"]


def test_kept_verdicts_are_only_those_the_last_run_used(monkeypatch, tmp_path):
    calls = []
    _counting_validator(monkeypatch, calls, {"dxc": "dxcompiler.dll: 1.8"})
    monkeypatch.setenv(runner.VERDICTS, str(tmp_path))
    runner.run(["good", "bad"], workers=1)
    assert len(list(tmp_path.iterdir())) == 2
    # Kept verdicts carry the time of the run that wrote them.
    for path in tmp_path.iterdir():
        os.utime(path, (1, 1))
    runner.run(["good"], workers=1)
    assert len(list(tmp_path.iterdir())) == 1
    runner.run(["bad"], workers=1)
    assert calls == ["good", "bad", "bad"]


def test_kept_folders_are_file_names_that_two_features_never_share():
    from tools.coverage.model import kept_folder

    assert kept_folder("glsl", "input", "abs") == "glsl-input/abs"
    rust = kept_folder("rust", "output", "Float::abs")
    assert (
        rust.startswith("rust-output/Float_abs-")
        and len(rust) == len("rust-output/Float_abs-") + 8
    )
    assert rust != kept_folder("rust", "output", "Float:abs")


def test_a_kept_run_writes_each_probe_and_its_translations(monkeypatch, tmp_path):
    from tools.coverage.model import Param

    vec3 = Type("float", (3,))
    feature = Feature(
        "glsl", "function", "abs", Signature(vec3, (Param(vec3),)), crossgl="abs"
    )
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(
        validators,
        "validate",
        lambda language, source, stage="compute", entry="", target="": validators.Verdict(
            True, ""
        ),
    )
    stale = tmp_path / "glsl-output" / "abs" / "translated.frag"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier run")
    results = runner.run(
        [(feature, "output", "glsl"), (feature, "input", "glsl")],
        workers=1,
        keep=tmp_path,
    )
    assert [r.outcome for r in results] == ["supported", "supported"]
    kept = {
        str(path.relative_to(tmp_path)): path.read_text()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert sorted(kept) == [
        "glsl-input/abs/crossgl.cgl",
        "glsl-input/abs/probe.comp",
        "glsl-input/abs/translated.comp",
        "glsl-output/abs/native.comp",
        "glsl-output/abs/probe.cgl",
        "glsl-output/abs/translated.comp",
    ]
    assert "abs(" in kept["glsl-output/abs/probe.cgl"]
    assert "#version" in kept["glsl-output/abs/translated.comp"]
    assert runner._kept is None


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


def _hlsl_row(*features, **row):
    return {
        "language": "hlsl",
        "direction": "output",
        "offered": True,
        "executes": True,
        **row,
        "features": [
            {
                "name": name,
                "category": "function",
                "group": "g",
                "outcome": outcome,
                **extra,
            }
            for name, outcome, extra in features
        ],
    }


def test_merge_adds_the_execution_verdicts_of_an_execution_only_report():
    lavapipe = {"runtime": "vulkan-lavapipe"}
    diverged = {"runtime": "vulkan-lavapipe", "detail": "on vulkan-lavapipe, diverged"}
    linux = _report(
        _hlsl_row(
            ("abs", "supported", lavapipe),
            ("cos", "supported", lavapipe),
            # lavapipe skips a probe that rounds through a 16-bit float; WARP runs it.
            ("f16tof32", "supported", {}),
            ("sin", "wrong", diverged),
            ("tan", "refused", {"detail": "refused"}),
            ("exp", "supported", lavapipe),
            executed=4,
            runtime="vulkan-lavapipe",
            executions={"vulkan-lavapipe": 4},
        ),
        tool="dxc",
    )
    warp = {"runtime": "d3d12-warp"}
    windows = _report(
        _hlsl_row(
            ("abs", "supported", warp),
            ("cos", "wrong", {**warp, "detail": "on d3d12-warp, diverged"}),
            ("f16tof32", "supported", warp),
            ("sin", "supported", warp),
            # Its static verdict differs; the Linux report decides static verdicts.
            ("tan", "supported", warp),
            ("exp", "wrong", {"detail": "rejected by DXC"}),
            executed=5,
            runtime="d3d12-warp",
        ),
        tool="dxc",
    )
    windows["execution_only"] = True
    windows["validators"] = {"dxc": "windows"}
    merged = report.merge([windows, linux])
    assert "execution_only" not in merged and merged["validators"] == {"dxc": "1.0"}
    [row] = [b for b in merged["backends"] if b["language"] == "hlsl"]
    found = {
        f["name"]: (f["outcome"], f.get("runtime"), f.get("detail"))
        for f in row["features"]
    }
    both = "d3d12-warp, vulkan-lavapipe"
    assert found == {
        "abs": ("supported", both, None),
        "cos": ("wrong", both, "on d3d12-warp, diverged"),
        "f16tof32": ("supported", "d3d12-warp", None),
        "sin": ("wrong", both, "on vulkan-lavapipe, diverged"),
        "tan": ("refused", None, "refused"),
        "exp": ("supported", "vulkan-lavapipe", None),
    }
    assert row["counts"] == {"supported": 3, "refused": 1, "wrong": 2, "untested": 0}
    assert row["score"] == 50.0
    assert row["executed"] == 5 and row["runtime"] == both
    assert row["executions"] == {"d3d12-warp": 4, "vulkan-lavapipe": 4}
    assert "| 4 on d3d12-warp, 4 on vulkan-lavapipe |" in publish.markdown(merged)


def test_merge_lists_the_probes_an_execution_only_runtime_did_not_run():
    half = {"unrun": ["on vulkan-lavapipe, its SPIR-V declares a 16-bit float"]}
    linux = _report(
        _hlsl_row(
            ("abs", "supported", {"runtime": "vulkan-lavapipe"}),
            ("f16tof32", "supported", half),
            ("tan", "refused", {"detail": "refused"}),
            executed=1,
            runtime="vulkan-lavapipe",
        ),
        tool="dxc",
    )
    later = "on d3d12-warp, its native probe did not run (invalid)\nerror: needs shader model 6.9"
    windows = _report(
        _hlsl_row(
            ("abs", "supported", {"unrun": [later]}),
            ("f16tof32", "supported", {"unrun": [later]}),
            # A refused probe stays refused, with no reason to run it.
            ("tan", "supported", {"unrun": [later]}),
        ),
        tool="dxc",
    )
    windows["execution_only"] = True
    [row] = [
        b for b in report.merge([linux, windows])["backends"] if b["language"] == "hlsl"
    ]
    found = {f["name"]: f.get("unrun") for f in row["features"]}
    assert found == {"abs": [later], "f16tof32": [*half["unrun"], later], "tan": None}
    text = publish.markdown(report.merge([linux, windows]))
    assert "<summary>2 supported compute probes a runtime did not run</summary>" in text
    assert (
        "- `on d3d12-warp, its native probe did not run (invalid)` (2): `abs`, `f16tof32`"
        in text
    )
    assert (
        "- `on vulkan-lavapipe, its SPIR-V declares a 16-bit float`: `f16tof32`" in text
    )
    assert "needs shader model 6.9" in report.dumps(report.merge([linux, windows]))


def test_merge_needs_a_full_report_for_every_execution_only_row():
    windows = _report(_hlsl_row(("abs", "supported", {"runtime": "d3d12-warp"})))
    windows["execution_only"] = True
    with pytest.raises(ValueError, match="every report is execution-only"):
        report.merge([windows])
    linux = _report(_row("glsl", "output", "supported"))
    with pytest.raises(ValueError, match="hlsl output has execution verdicts"):
        report.merge([linux, windows])


def test_execution_only_needs_execute(capsys):
    from tools.coverage import __main__ as cli

    assert cli.main(["run", "--backend", "hlsl", "--execution-only"]) == 2
    assert "--execution-only needs --execute" in capsys.readouterr().err


def test_run_measures_only_the_requested_features(monkeypatch, tmp_path, capsys):
    from tools.coverage import __main__ as cli
    from tools.coverage import features

    other = Feature("glsl", "function", "sin", ABS.signature, crossgl="sin")
    monkeypatch.setattr(features, "load", lambda language, offline=False: [ABS, other])
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(lowering, "can_lower", lambda language, target="": True)
    measured = []

    def run(jobs, workers=None, keep=None):
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
    monkeypatch.setattr(lowering, "can_lower", lambda language: language == "glsl")
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
    monkeypatch.setattr(lowering, "can_lower", lambda language: False)
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
