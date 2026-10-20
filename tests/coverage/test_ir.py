from tools.coverage import ir, report
from tools.coverage.model import Feature, Result, Signature, Type

FLOAT = Type("float")


def row(language, *features):
    return {
        "language": language,
        "direction": "output",
        "offered": True,
        "features": [
            {"name": name, "category": "function", "outcome": outcome, **extra}
            for name, outcome, extra in features
        ],
    }


def by_name(entry):
    return {feature["name"]: feature for feature in entry["features"]}


def test_members_with_one_semantic_id_form_one_feature():
    entry = ir.build(
        [
            row("glsl", ("fma", "supported", {"semantic": "GLSL.std.450:Fma"})),
            row(
                "hlsl",
                ("mad", "wrong", {"semantic": "GLSL.std.450:Fma", "detail": "bad"}),
            ),
        ]
    )
    fma = by_name(entry)["GLSL.std.450:Fma"]
    assert fma["outcome"] == "supported"
    assert fma["via"] == ["glsl"]
    assert fma["members"] == ["glsl/fma", "hlsl/mad"]
    assert fma["group"] == "shared"
    assert (entry["language"], entry["direction"], entry["score"]) == (
        "crossgl",
        "ir",
        100.0,
    )


def test_unmatched_features_stay_their_own_and_wrong_outranks_refused():
    entry = ir.build(
        [
            row(
                "hlsl",
                ("abort", "untested", {"detail": "no probe"}),
                ("Op", "refused", {"semantic": "OpX", "detail": "gap"}),
            ),
            row("spirv", ("X", "wrong", {"semantic": "OpX", "detail": "invalid"})),
        ]
    )
    features = by_name(entry)
    assert features["hlsl/abort"]["outcome"] == "untested"
    assert features["hlsl/abort"]["group"] == "hlsl"
    assert features["OpX"]["outcome"] == "wrong"
    assert features["OpX"]["detail"] == "spirv: invalid"
    assert entry["counts"] == {"supported": 0, "refused": 0, "wrong": 1, "untested": 1}


def test_input_rows_do_not_widen_the_universe():
    entry = ir.build([{**row("glsl", ("abs", "supported", {})), "direction": "input"}])
    assert entry is None


def test_report_adds_the_ir_row_after_the_backends():
    feature = Feature("glsl", "function", "abs", Signature(FLOAT, ()), crossgl="abs")
    results = [Result(feature, "output", "supported", semantic="GLSL.std.450:FAbs")]
    data = report.build(results, ["glsl"], ["output"])
    assert [b["language"] for b in data["backends"]] == ["glsl", "crossgl"]
    assert data["backends"][0]["features"][0]["semantic"] == "GLSL.std.450:FAbs"
    assert "CrossGL IR" in report.summary(data)
