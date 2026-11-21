"""Measuring the translation matrix: planning the cells from a report, running them, and reporting and merging them."""

import json
from dataclasses import replace

import numpy as np
import pytest

from tools.coverage import __main__ as cli
from tools.coverage import (
    execution,
    features,
    lowering,
    pairs,
    report,
    runner,
    validators,
)
from tools.coverage.model import PAIR, Feature, Param, Result, Signature, Type
from tools.coverage.upstream import UpstreamError

VEC3 = Type("float", (3,))
ABS = Signature(VEC3, (Param(VEC3),))
FMA = {"semantic": "GLSL.std.450:Fma"}
BARRIER = {"semantic": "construct:compute/barrier-in-loop"}


def row(language, direction, *features, offered=True):
    return {
        "language": language,
        "direction": direction,
        "offered": offered,
        "features": [
            {"name": name, "category": category, "outcome": outcome, **extra}
            for name, category, outcome, extra in features
        ],
    }


def rows():
    return [
        row("glsl", "output", ("fma", "function", "supported", FMA)),
        row(
            "glsl",
            "input",
            ("fma", "function", "untested", {"detail": "the probe itself is rejected"}),
        ),
        row(
            "hlsl",
            "output",
            ("mad", "function", "supported", FMA),
            ("barrier-in-loop", "compute", "supported", BARRIER),
        ),
        row(
            "hlsl",
            "input",
            ("mad", "function", "supported", {}),
            ("barrier-in-loop", "compute", "wrong", {}),
        ),
        row(
            "webgl",
            "output",
            ("fma", "function", "supported", FMA),
            ("barrier-in-loop", "compute", "supported", BARRIER),
        ),
    ]


CATALOG = {
    "glsl": [Feature("glsl", "function", "fma", ABS)],
    "hlsl": [
        Feature("hlsl", "function", "mad", ABS),
        Feature("hlsl", "compute", "barrier-in-loop", None),
    ],
    "webgl": [
        Feature("webgl", "function", "fma", ABS, stage="fragment"),
        Feature("webgl", "compute", "barrier-in-loop", None, stage="fragment"),
    ],
}


def test_a_plan_measures_the_cells_into_the_requested_targets():
    measured = pairs.plan(rows(), ["glsl", "webgl"], CATALOG.__getitem__)
    assert measured.cells == [("glsl", "webgl"), ("hlsl", "glsl"), ("hlsl", "webgl")]
    assert [(job[0].key, job[1], job[2], job[3].key) for job in measured.jobs] == [
        ("hlsl/function/mad", PAIR, "glsl", "glsl/function/fma")
    ]


def test_a_feature_its_source_row_leaves_untested_is_untested_in_its_cells():
    measured = pairs.plan(rows(), ["hlsl"], CATALOG.__getitem__)
    assert measured.jobs == []
    [untested] = measured.results
    assert (untested.outcome, untested.into, untested.direction) == (
        "untested",
        "hlsl",
        PAIR,
    )
    assert (
        untested.detail
        == "untested in the glsl input row: the probe itself is rejected"
    )


def test_a_compute_probe_is_left_out_of_a_cell_into_webgl():
    measured = pairs.plan(rows(), ["webgl"], CATALOG.__getitem__)
    # HLSL's compute probes have no WebGL translation, so nothing of HLSL is measured into WebGL.
    assert not [job for job in measured.jobs if job[0].language == "hlsl"]
    assert ("hlsl", "webgl") in measured.cells


def test_a_plan_leaves_out_features_the_catalog_no_longer_lists_and_unnamed_ones():
    catalog = {**CATALOG, "hlsl": CATALOG["hlsl"][1:]}
    assert not pairs.plan(rows(), ["glsl"], catalog.__getitem__).jobs
    named = pairs.plan(rows(), ["glsl"], CATALOG.__getitem__, names=["other"])
    assert not named.jobs and named.cells == [("hlsl", "glsl")]


def test_the_runner_runs_a_pair_job_and_clears_its_kept_folder(monkeypatch, tmp_path):
    stale = tmp_path / "hlsl-to-glsl" / "mad" / "old.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier run")
    calls = []

    def run_pair(feature, source, target, partner):
        calls.append((feature.name, source, target, partner.name))
        return Result(feature, PAIR, "supported", into=target)

    monkeypatch.setattr(pairs, "run_pair", run_pair)
    mad, fma = CATALOG["hlsl"][0], CATALOG["glsl"][0]
    [result] = runner.run([(mad, PAIR, "glsl", fma)], workers=1, keep=tmp_path)
    assert calls == [("mad", "hlsl", "glsl", "fma")]
    assert result.into == "glsl"
    assert not stale.exists()


def _results():
    mad, barrier = CATALOG["hlsl"]
    return [
        Result(mad, PAIR, "supported", validator="glslangValidator", into="glsl"),
        Result(barrier, PAIR, "refused", "E1: no barrier", into="glsl"),
        Result(mad, "output", "supported"),
    ]


def test_a_report_scores_each_cell_like_a_row():
    data = report.build(
        _results(), ["glsl"], [], cells=[("hlsl", "glsl"), ("cuda", "glsl")]
    )
    assert data["backends"] == []
    cuda, hlsl = data["pairs"]
    assert (cuda["source"], cuda["score"], cuda["features"]) == ("cuda", None, [])
    assert (hlsl["source"], hlsl["target"], hlsl["score"]) == ("hlsl", "glsl", 50.0)
    assert hlsl["counts"] == {"supported": 1, "refused": 1, "wrong": 0, "untested": 0}
    assert hlsl["executes"] is True
    assert hlsl["features"][0] == {
        "name": "barrier-in-loop",
        "category": "compute",
        "outcome": "refused",
        "detail": "E1: no barrier",
    }
    assert report.summary(data).splitlines()[-1].split() == [
        *("hlsl", "to", "glsl", "50.0%"),
        *("1", "1", "0", "0"),
    ]


def test_a_report_without_cells_has_no_pairs():
    assert "pairs" not in report.build(_results(), ["glsl"], [])


def test_cells_survive_the_stored_form():
    data = report.build(_results(), ["glsl"], [], cells=[("hlsl", "glsl")])
    stored = json.loads(report.dumps(data))
    assert stored["pairs"][0]["columns"] == ["name", "category", "outcome", "detail"]
    assert report.loads(report.dumps(data))["pairs"] == data["pairs"]


def test_merge_adds_the_cells_of_each_report_to_the_rows():
    rows_report = report.build(_results(), ["hlsl"], ["output"])
    into_glsl = report.build(_results(), ["glsl"], [], cells=[("hlsl", "glsl")])
    into_metal = report.build([], ["metal"], [], cells=[("hlsl", "metal")])
    merged = report.merge([rows_report, into_metal, into_glsl])
    assert [(c["source"], c["target"]) for c in merged["pairs"]] == [
        ("hlsl", "glsl"),
        ("hlsl", "metal"),
    ]
    # The cells stay apart from the rows, so the CrossGL IR row is built from the rows alone.
    assert [b["language"] for b in merged["backends"]] == ["hlsl", "crossgl"]
    assert "pairs" not in report.merge([rows_report])


def test_merge_rejects_a_cell_measured_twice():
    cell = report.build(_results(), ["glsl"], [], cells=[("hlsl", "glsl")])
    with pytest.raises(ValueError, match="hlsl to glsl is measured by more than one"):
        report.merge([cell, cell])


GLSL_SOURCE = "layout(std430) buffer B { float fin[]; }; void main() { fout = fin[0]; }"


def test_only_cells_whose_ends_lower_to_spirv_run():
    assert execution.runs_pair("hlsl", "glsl")
    assert execution.runs_pair("glsl", "spirv")
    assert not execution.runs_pair("metal", "hlsl")
    assert not execution.runs_pair("hlsl", "cuda")
    assert not execution.runs_pair("glsl", "glsl")
    assert execution.as_spirv("spirv", "OpCapability Shader") == "OpCapability Shader"


def test_a_cell_lowers_each_source_with_its_own_language(monkeypatch, tmp_path):
    lowered = []

    def as_spirv(language, source):
        lowered.append(language)
        return ""

    monkeypatch.setattr(execution, "as_spirv", as_spirv)
    mad = CATALOG["hlsl"][0]
    result = Result(mad, PAIR, "supported", sources=("a", "b"), into="glsl")
    assert execution._execute("hlsl", [(0, result)], {}, tmp_path, True) == {}
    assert lowered == ["hlsl", "glsl"]


def test_run_executes_a_cell_as_spirv_and_skips_one_on_two_runtimes(monkeypatch):
    passes = []

    def run_pass(language, items, runtimes, root, workers, via_spirv):
        passes.append((language, [r.into for _, r in items], root.name, via_spirv))
        return {}, ""

    monkeypatch.setattr(execution, "_pass", run_pass)
    mad = CATALOG["hlsl"][0]
    supported = Result(mad, PAIR, "supported", sources=(GLSL_SOURCE,) * 2)
    cells = [replace(supported, into="glsl"), replace(supported, into="metal")]
    results, used = execution.run(cells)
    assert passes == [("hlsl", ["glsl"], "hlsl-to-glsl", True)]
    assert (results, used) == (cells, [])


def test_a_diverging_cell_names_both_compilers():
    mad = CATALOG["hlsl"][0]
    result = Result(mad, PAIR, "supported", into="glsl")
    native = {"outcome": "preserved", "runtime": "vulkan-lavapipe"}
    outputs = {
        "00000-native": np.array([1, 2], dtype=np.uint32),
        "00000-translated": np.array([1, 3], dtype=np.uint32),
    }
    judged = execution._judge(result, native, dict(native), outputs, False, "", True, 0)
    assert judged.outcome == "wrong"
    assert judged.detail.startswith(
        "on vulkan-lavapipe as SPIR-V from dxc and glslangValidator, the translation "
        + execution.DIVERGED
    )
    assert execution.found(judged.detail)


def test_idle_names_a_cell_that_ran_nothing():
    mad = CATALOG["hlsl"][0]
    ready = Result(mad, PAIR, "supported", sources=(GLSL_SOURCE,) * 2, into="glsl")
    assert execution.idle([ready]) == ["hlsl to glsl"]
    assert execution.idle([replace(ready, runtime="vulkan-lavapipe")]) == []


def _loaded(language):
    try:
        return {f.name: f for f in features.load(language, offline=True)}
    except UpstreamError:
        pytest.skip(f"the {language} feature sources are not cached")


def _spirv_runtime():
    try:
        from tests.conformance import runner as harness

        selection = harness.select_runtimes("available", ["spirv"])
    except Exception:
        return False
    return any(available.available for _, available in selection.get("spirv", []))


@pytest.mark.skipif(
    not (validators.available("spirv") and lowering.can_lower("glsl")),
    reason="spirv-val, glslang or spirv-dis is not installed",
)
@pytest.mark.skipif(not _spirv_runtime(), reason="no Vulkan runtime")
def test_a_glsl_probe_translated_to_spirv_computes_what_the_native_probe_computes():
    glsl, spirv = _loaded("glsl"), _loaded("spirv")
    result = pairs.run_pair(glsl["floor"], "glsl", "spirv", spirv["Floor"])
    assert result.outcome == "supported", result.detail
    native, translated = result.sources
    assert "Floor" in translated
    other = replace(result, sources=(native, translated.replace("Floor", "Ceil")))
    (same, diverged), used = execution.run([result, other], workers=1)
    assert used and same.outcome == "supported" and same.runtime
    assert diverged.outcome == "wrong"
    assert execution.DIVERGED in diverged.detail
    assert "as SPIR-V from glslangValidator," in diverged.detail


def test_the_cli_measures_the_cells_a_report_defines(monkeypatch, tmp_path):
    source = tmp_path / "rows.json"
    report.write({**report.build([], [], []), "backends": rows()}, source)
    monkeypatch.setattr(
        features, "load", lambda language, offline=False: CATALOG[language]
    )
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    jobs = []

    def run(planned, workers=None, keep=None):
        jobs.extend(planned)
        return [Result(job[0], PAIR, "supported", into=job[2]) for job in planned]

    monkeypatch.setattr(runner, "run", run)
    output = tmp_path / "pairs.json"
    status = cli.main(
        ["run", "--pairs", str(source), "--backend", "glsl", "--output", str(output)]
    )
    assert status == 0
    assert [(job[0].name, job[2]) for job in jobs] == [("mad", "glsl")]
    [cell] = report.load(output)["pairs"]
    assert (cell["source"], cell["target"], cell["score"]) == ("hlsl", "glsl", 100.0)


def test_the_cli_refuses_a_direction_with_pairs(tmp_path, capsys):
    status = cli.main(
        ["run", "--pairs", str(tmp_path / "x.json"), "--direction", "input"]
    )
    assert status == 2
    assert "no direction" in capsys.readouterr().err
