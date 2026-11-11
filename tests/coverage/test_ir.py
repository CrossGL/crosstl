import pytest

from tools.coverage import causes, ir, report, runner, validators
from tools.coverage.model import Feature, Param, Result, Signature, Type

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


def test_a_call_that_spells_its_generic_arguments_is_keyed_by_its_name():
    entry = ir.build(
        [row("slang", ("bit_cast", "supported", {"crossgl": "bit_cast<int>"}))]
    )
    assert list(by_name(entry)) == ["slang/bit_cast"]


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


def test_a_feature_with_no_id_borrows_the_glsl_id_of_its_crossgl_spelling():
    entry = ir.build(
        [
            row(
                "glsl",
                ("frexp", "supported", {"semantic": "GLSL.std.450:Frexp"}),
            ),
            row(
                "wgsl",
                ("frexp", "wrong", {"crossgl": "frexp", "detail": "bad"}),
            ),
        ]
    )
    features = by_name(entry)
    assert list(features) == ["GLSL.std.450:Frexp"]
    assert features["GLSL.std.450:Frexp"]["members"] == ["glsl/frexp", "wgsl/frexp"]


def test_a_function_with_no_id_takes_the_one_id_of_functions_spelled_alike():
    cospi = "GLSL.std.450:Cos+OpVectorTimesScalar"
    method = row("metal", ("texture2d.read", "supported", {"crossgl": "Load"}))
    method["features"][0]["category"] = "method"
    entry = ir.build(
        [
            row(
                "slang",
                ("cospi", "refused", {"semantic": cospi, "detail": "gap"}),
                ("log10", "supported", {"semantic": "GLSL.std.450:Log+OpFMul"}),
                ("Load", "supported", {"semantic": "OpImageFetch"}),
            ),
            row(
                "hlsl",
                ("log10", "supported", {"semantic": "GLSL.std.450:Log2+OpFMul"}),
            ),
            row(
                "cuda",
                ("cospif", "supported", {"crossgl": "cospi"}),
                ("log10f", "supported", {"crossgl": "log10"}),
            ),
            method,
        ]
    )
    features = by_name(entry)
    assert features[cospi]["members"] == ["cuda/cospif", "slang/cospi"]
    assert features[cospi]["outcome"] == "supported"
    # Two ids share the spelling log10, so neither is taken; a method takes no function's id.
    assert features["cuda/log10"]["members"] == ["cuda/log10f"]
    assert features["metal/Load"]["members"] == ["metal/texture2d.read"]


def test_a_gap_names_its_member_as_the_feature_of_its_cause():
    detail = 'probe.cu(9): error: no instance of overloaded function "{}" matches'
    entry = ir.build(
        [
            row(
                "cuda",
                *(
                    (name, "wrong", {"detail": detail.format(name)})
                    for name in ("atomicAdd_block", "atomicOr_block")
                ),
            ),
        ]
    )
    groups = causes.grouped(entry)
    assert [(text, [f["name"] for f in features]) for text, features in groups] == [
        (
            'cuda: probe.cu(N): error: no instance of overloaded function "<feature>" matches',
            ["cuda/atomicAdd_block", "cuda/atomicOr_block"],
        )
    ]


def test_a_member_its_generator_fails_counts_when_spirv_expresses_it():
    expm1 = "GLSL.std.450:Exp+OpFSub"
    entry = ir.build(
        [
            row(
                "rust",
                (
                    "Float::exp_m1",
                    "wrong",
                    {"semantic": expm1, "detail": "no std", "expressed": "spirv"},
                ),
                ("Float::ln_1p", "wrong", {"semantic": "OpY", "detail": "no std"}),
            ),
        ]
    )
    features = by_name(entry)
    assert features[expm1]["outcome"] == "supported"
    assert features[expm1]["via"] == ["spirv"]
    assert "detail" not in features[expm1]
    # Without the route the generator's failure is still the IR gap.
    assert features["OpY"]["outcome"] == "wrong"
    assert features["OpY"]["detail"] == "rust: no std"


def test_the_report_keeps_the_route_of_a_result_that_is_not_supported():
    feature = Feature(
        "rust", "function", "Float::exp_m1", Signature(FLOAT, ()), crossgl="expm1"
    )
    result = Result(
        feature,
        "output",
        "wrong",
        "no std",
        semantic="GLSL.std.450:Exp+OpFSub",
        expressed="spirv",
    )
    data = report.loads(report.dumps(report.build([result], ["rust"], ["output"])))
    rust, crossgl = data["backends"]
    assert rust["features"][0]["expressed"] == "spirv"
    assert rust["score"] == 0.0
    assert crossgl["score"] == 100.0


RUST_ABS = Feature(
    "rust",
    "function",
    "Float::abs",
    Signature(FLOAT, (Param(FLOAT),)),
    group="float",
    crossgl="abs",
)


@pytest.mark.skipif(
    not validators.available("spirv"), reason="spirv-val is not installed"
)
def test_a_crossgl_probe_whose_spirv_has_the_native_id_is_expressed():
    assert runner.expressed(RUST_ABS, "rust", "GLSL.std.450:FAbs") == "spirv"
    # The translation's call must lower to the native probe's id, not merely validate.
    assert runner.expressed(RUST_ABS, "rust", "GLSL.std.450:FSign") == ""


@pytest.mark.skipif(
    not validators.available("spirv"), reason="spirv-val is not installed"
)
def test_a_rust_subgroup_function_is_expressed_through_its_wave_spelling():
    uint = Type("uint")
    add = Feature(
        "rust",
        "function",
        "subgroup_i_add",
        Signature(uint, (Param(uint),)),
        group="arch",
        crossgl="WaveActiveSum",
    )
    # The operand type selects the integer instruction rust-gpu's native probe lowers to.
    assert runner.expressed(add, "rust", "OpGroupNonUniformIAdd.Reduce") == "spirv"
    assert runner.expressed(add, "rust", "OpGroupNonUniformFAdd.Reduce") == ""


def test_no_route_for_spirv_itself_constructs_or_features_without_an_id():
    assert runner.expressed(RUST_ABS, "spirv", "GLSL.std.450:FAbs") == ""
    assert runner.expressed(RUST_ABS, "rust", "") == ""
    assert runner.expressed(RUST_ABS, "rust", "construct:control-flow/loop") == ""


def test_only_output_results_that_are_not_supported_look_for_the_route(monkeypatch):
    asked = []
    monkeypatch.setattr(
        runner,
        "expressed",
        lambda feature, language, semantic: asked.append(language) or "spirv",
    )
    for outcome in ("supported", "wrong", "refused", "untested"):
        monkeypatch.setattr(
            runner,
            "run_output",
            lambda feature, language, outcome=outcome: Result(
                feature, "output", outcome, semantic="GLSL.std.450:FAbs"
            ),
        )
        result = runner._job((RUST_ABS, "output", "rust"))
        assert result.expressed == ("spirv" if outcome in ("wrong", "refused") else "")
    assert asked == ["rust", "rust"]
