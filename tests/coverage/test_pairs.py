import pytest

from tools.coverage import ir, pairs, runner, validators
from tools.coverage.model import Feature, Param, Signature, Type

VEC3 = Type("float", (3,))
ABS = Signature(VEC3, (Param(VEC3),))


def row(language, direction, *features, offered=True):
    return {
        "language": language,
        "direction": direction,
        "offered": offered,
        "features": [
            {"name": name, "category": category, "outcome": "supported", **extra}
            for name, category, extra in features
        ],
    }


def both(language, *features):
    return [row(language, "output", *features), row(language, "input", *features)]


FMA = {"semantic": "GLSL.std.450:Fma"}
BARRIER = {"semantic": "construct:compute/barrier-in-loop"}


def backends():
    return [
        *both(
            "glsl",
            ("fma", "function", FMA),
            ("barrier-in-loop", "compute", BARRIER),
            ("dFdx", "function", {"semantic": "OpDPdx"}),
        ),
        *both(
            "hlsl",
            ("mad", "function", FMA),
            ("barrier-in-loop", "compute", BARRIER),
            ("abort", "function", {}),
        ),
        row(
            "opencl",
            "input",
            ("fma", "function", {}),
            ("barrier-in-loop", "compute", {}),
            ("get_global_id", "function", {"crossgl": "gl_GlobalInvocationID"}),
        ),
        row("wgsl", "input", ("fma", "function", {}), offered=False),
    ]


def test_keys_match_the_ir_row_and_key_opencl_by_construct_and_spelling():
    keyed = pairs.keys(backends())
    for key, members in ir.groups(backends()).items():
        for language, feature in members:
            assert keyed[language, feature["category"], feature["name"]] == key
    assert keyed["opencl", "function", "fma"] == "GLSL.std.450:Fma"
    assert keyed["opencl", "compute", "barrier-in-loop"] == BARRIER["semantic"]
    # A spelling no output function shares is a key of its own.
    assert keyed["opencl", "function", "get_global_id"] == (
        "opencl/gl_GlobalInvocationID"
    )
    assert not any(language == "wgsl" for language, _, _ in keyed)


def test_an_ambiguous_spelling_keeps_an_opencl_function_apart():
    rows = [
        *both("glsl", ("max", "function", {"semantic": "GLSL.std.450:FMax"})),
        *both("metal", ("max", "function", {"semantic": "GLSL.std.450:NMax"})),
        row("opencl", "input", ("max", "function", {})),
    ]
    assert pairs.keys(rows)["opencl", "function", "max"] == "opencl/max"


def test_cells_hold_the_shared_features_of_every_pair_but_the_input_rows():
    matrix = pairs.cells(backends())
    assert sorted(matrix) == [
        ("glsl", "hlsl"),
        ("hlsl", "glsl"),
        ("opencl", "glsl"),
    ]
    assert matrix["glsl", "hlsl"] == [
        pairs.Member(
            "compute",
            "barrier-in-loop",
            BARRIER["semantic"],
            (("compute", "barrier-in-loop"),),
        ),
        pairs.Member("function", "fma", FMA["semantic"], (("function", "mad"),)),
    ]
    # dFdx and abort have no partner in the other language, so they are outside the cells.
    assert [m.name for m in matrix["hlsl", "glsl"]] == ["barrier-in-loop", "mad"]
    assert [m.partners for m in matrix["opencl", "glsl"]] == [
        (("compute", "barrier-in-loop"),),
        (("function", "fma"),),
    ]


def test_input_rows_are_not_cells():
    assert pairs.is_row("glsl", "glsl")
    assert pairs.is_row("opencl", "hlsl")
    assert not pairs.is_row("opencl", "glsl")
    assert not pairs.is_row("hlsl", "glsl")


def test_cells_list_pairs_that_share_nothing():
    rows = [
        *both("glsl", ("dFdx", "function", {"semantic": "OpDPdx"})),
        *both("cuda", ("__clz", "function", {})),
    ]
    assert pairs.cells(rows) == {("cuda", "glsl"): [], ("glsl", "cuda"): []}


def test_a_compute_probe_is_outside_webgl():
    compute = Feature("glsl", "function", "abs", ABS)
    fragment = Feature("glsl", "graphics", "flat-input", None, stage="fragment")
    assert pairs.outside(compute, "webgl") == "GLSL ES 3.00 has no compute stage"
    assert pairs.outside(fragment, "webgl") == ""
    assert pairs.outside(compute, "hlsl") == ""


def test_the_partner_prefers_the_usual_compiler_target():
    usual = Feature("glsl", "function", "subgroupAdd", ABS)
    vulkan = Feature("glsl", "function", "subgroupAdd2", ABS, target="vulkan")
    assert pairs.choose_partner([vulkan, usual]) is usual
    assert pairs.choose_partner([vulkan]) is vulkan


@pytest.fixture
def checked(monkeypatch):
    calls = []

    def validate(language, source, stage="compute", entry="", target=""):
        calls.append((language, stage, entry, target))
        return validators.Verdict(True, "")

    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(validators, "validate", validate)
    return calls


GLSL_ABS = Feature("glsl", "function", "abs", ABS, crossgl="abs")
HLSL_ABS = Feature("hlsl", "function", "abs", ABS, crossgl="abs")


def test_a_pair_translates_the_native_probe_straight_to_the_target(checked):
    result = pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS)
    assert (result.outcome, result.direction, result.validator, result.into) == (
        "supported",
        "pair",
        validators.TOOLS["hlsl"],
        "hlsl",
    )
    # The target's compiler checks its own probe of the partner, then the translation.
    assert checked == [("hlsl", "compute", "main", ""), ("hlsl", "compute", "", "")]


@pytest.mark.skipif(not validators.available("glsl"), reason="glslang is not installed")
def test_an_hlsl_probe_translates_to_glsl_that_glslang_accepts():
    result = pairs.run_pair(HLSL_ABS, "hlsl", "glsl", GLSL_ABS)
    assert result.outcome == "supported", result.detail
    native, translated = result.sources
    assert "abs(" in native and "abs(" in translated


def test_a_pair_is_one_crosstl_call_from_the_source_backend(monkeypatch, checked):
    calls = []

    def translate(source, suffix, backend, source_backend):
        calls.append((suffix, backend, source_backend))
        return "[numthreads(1, 1, 1)] void main() {}"

    monkeypatch.setattr(runner, "translate", translate)
    assert pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS).outcome == "supported"
    assert calls == [(".comp", "directx", "opengl")]


def test_a_webgl_probe_of_a_compute_feature_is_checked_as_a_fragment_shader(
    monkeypatch, checked
):
    # GLSL ES 3.00 has no compute stage, so the WebGL probe and its translation are fragment shaders.
    monkeypatch.setattr(runner, "translate", lambda *args: "void main() {}")
    webgl = Feature("webgl", "function", "abs", ABS, crossgl="abs")
    result = pairs.run_pair(webgl, "webgl", "glsl", GLSL_ABS)
    assert result.outcome == "supported"
    assert result.sources == ()
    assert checked == [("glsl", "compute", "main", ""), ("glsl", "fragment", "", "")]


def test_the_partner_sets_the_compiler_target(checked):
    vulkan = Feature("glsl", "function", "abs", ABS, crossgl="abs", target="vulkan")
    pairs.run_pair(HLSL_ABS, "hlsl", "glsl", vulkan)
    assert [call[3] for call in checked] == ["vulkan", "vulkan"]


@pytest.mark.parametrize(
    ("error", "outcome"), [(runner.Refused, "refused"), (runner.Broken, "wrong")]
)
def test_a_refusal_or_crash_is_classified_without_the_target_tool(
    monkeypatch, error, outcome
):
    def translate(*args):
        raise error("E123: no translation")

    monkeypatch.setattr(runner, "translate", translate)
    monkeypatch.setattr(validators, "available", lambda language, target="": False)
    result = pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS)
    assert (result.outcome, result.detail) == (outcome, "E123: no translation")


def test_a_missing_target_tool_leaves_a_translation_untested(monkeypatch):
    monkeypatch.setattr(validators, "available", lambda language, target="": False)
    result = pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS)
    assert result.outcome == "untested"
    assert result.detail.endswith("is not installed")
    assert result.validator == ""


def _rejecting(translation: bool):
    """A validator that rejects the translation, which is checked without an entry point, or the native probe."""

    def validate(language, source, stage="compute", entry="", target=""):
        rejected = (entry == "") == translation
        return validators.Verdict(not rejected, "error: bad" if rejected else "")

    return validate


def test_a_rejected_translation_is_wrong(monkeypatch):
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(validators, "validate", _rejecting(translation=True))
    result = pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS)
    assert result.outcome == "wrong"
    assert "bad" in result.detail


def test_a_feature_the_target_rejects_in_its_own_probe_is_untested(monkeypatch):
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(validators, "validate", _rejecting(translation=False))
    monkeypatch.setattr(
        runner, "translate", lambda *args: pytest.fail("translated a lacking feature")
    )
    result = pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS)
    assert result.outcome == "untested"
    assert result.detail == "the target's own probe is rejected: error: bad"


def test_a_placeholder_in_the_translation_is_wrong(monkeypatch, checked):
    monkeypatch.setattr(
        runner,
        "translate",
        lambda *args: "void main() { float x = /* abs not supported */ 0.0; }",
    )
    result = pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS)
    assert result.outcome == "wrong"
    assert "abs not supported" in result.detail


def test_a_feature_without_a_native_probe_is_untested():
    nameless = Feature("glsl", "function", "abs", None)
    result = pairs.run_pair(nameless, "glsl", "hlsl", HLSL_ABS)
    assert result.outcome == "untested"


def test_a_kept_pair_writes_its_probe_and_translation(checked, tmp_path):
    runner._remember(None, str(tmp_path))
    try:
        pairs.run_pair(GLSL_ABS, "glsl", "hlsl", HLSL_ABS)
    finally:
        runner._remember(None)
    kept = sorted(
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert kept == [
        "glsl-to-hlsl/abs/partner.hlsl",
        "glsl-to-hlsl/abs/probe.comp",
        "glsl-to-hlsl/abs/translated.hlsl",
    ]
