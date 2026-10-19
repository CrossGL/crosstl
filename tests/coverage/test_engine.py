import hashlib

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
    assert "not offered" in report.summary(data)


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
