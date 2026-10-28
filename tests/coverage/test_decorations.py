import pytest

from tools.coverage import decorations, validators
from tools.coverage.model import Feature

SAMPLE = Feature(
    "glsl",
    "graphics",
    "sample-interpolation",
    None,
    group="Decoration.Sample Capability.SampleRateShading",
    stage="fragment",
)
DEPTH = Feature(
    "hlsl",
    "graphics",
    "fragment-depth",
    None,
    group="BuiltIn.FragDepth",
    stage="fragment",
)
DECORATED = (
    "OpCapability Shader\nOpCapability SampleRateShading\n"
    "OpDecorate %uv Location 0\nOpDecorate %uv Sample\n"
)


def test_a_translation_must_keep_each_anchor_of_its_graphics_feature():
    assert decorations.lost(SAMPLE, "spirv", DECORATED) == ""
    assert decorations.lost(SAMPLE, "spirv", DECORATED.replace("Sample\n", "\n")) == (
        "the translation loses Decoration.Sample"
    )
    assert decorations.lost(SAMPLE, "spirv", "OpDecorate %uv Location 0\n") == (
        "the translation loses Decoration.Sample, Capability.SampleRateShading"
    )
    # A decoration may sit on a member of the interface block, and a capability name is matched whole.
    member = "OpCapability SampleRateShading\nOpMemberDecorate %Input 0 Sample\n"
    assert decorations.lost(SAMPLE, "spirv", member) == ""
    assert decorations.lost(
        SAMPLE, "spirv", member.replace(" 0 Sample", " 0 SampleX")
    ) == ("the translation loses Decoration.Sample")
    assert (
        decorations.lost(DEPTH, "spirv", "OpDecorate %depth BuiltIn FragDepth\n") == ""
    )
    assert decorations.lost(DEPTH, "spirv", "OpDecorate %depth Location 1\n") == (
        "the translation loses BuiltIn.FragDepth"
    )


def test_only_graphics_features_are_checked():
    loop = Feature("glsl", "control-flow", "for-loop", None, group="OpLoopMerge")
    shift = Feature("glsl", "operator", "vector-shift", None, group="Decoration.Flat")
    assert decorations.lost(loop, "spirv", "") == ""
    assert decorations.lost(shift, "spirv", "") == ""


def test_a_translation_is_lowered_with_its_entry_point_and_automatic_locations(
    monkeypatch,
):
    calls = []

    def lower(language, source, entry, stage, **options):
        calls.append((language, entry, stage, options))
        return "OpDecorate %depth BuiltIn FragDepth\n", ""

    monkeypatch.setattr(validators, "can_lower", lambda language: True)
    monkeypatch.setattr(validators, "lower", lower)
    hlsl = "struct Out { float d : SV_Depth; };\nOut PSMain() { Out o; o.d = 0.5; return o; }\n"
    assert decorations.lost(DEPTH, "hlsl", hlsl) == ""
    assert calls == [
        ("hlsl", "PSMain", "fragment", {"raw_ids": False, "auto_locations": True})
    ]


@pytest.mark.parametrize(
    "can_lower, lowered", [(False, ("", "")), (True, ("", "error: no location"))]
)
def test_a_translation_that_cannot_be_lowered_is_left_to_its_validator(
    monkeypatch, can_lower, lowered
):
    monkeypatch.setattr(validators, "can_lower", lambda language: can_lower)
    monkeypatch.setattr(validators, "lower", lambda *args, **kwargs: lowered)
    assert decorations.lost(DEPTH, "metal", "fragment float probe() {}") == ""
