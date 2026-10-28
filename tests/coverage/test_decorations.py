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
    assert decorations.lost(DEPTH, "hlsl", "float PSMain() : SV_Depth;") == ""


def test_opengl_glsl_is_lowered_with_the_vulkan_names_of_the_vertex_and_instance_index():
    source = "gl_Position = vec4(float(gl_VertexID + gl_InstanceID), 0.0, 0.0, 1.0);"
    assert validators.vulkan_glsl(source) == (
        "gl_Position = vec4(float(gl_VertexIndex + gl_InstanceIndex), 0.0, 0.0, 1.0);"
    )
    assert validators.vulkan_glsl("int gl_VertexIDs;") == "int gl_VertexIDs;"


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_an_opengl_vertex_index_read():
    source = (
        "#version 460\nlayout(location = 0) in vec4 position;\nvoid main() {\n"
        "    gl_Position = position + vec4(float(gl_VertexID), 0.0, 0.0, 0.0);\n}\n"
    )
    assembly, log = validators.lower("glsl", source, stage="vertex", raw_ids=False)
    assert "BuiltIn VertexIndex" in assembly, log


INVARIANT = Feature(
    "metal",
    "graphics",
    "invariant-position",
    None,
    group="Decoration.Invariant",
    stage="vertex",
)


def test_a_metal_translation_must_keep_the_air_attribute_of_each_anchor(monkeypatch):
    air = {"source": ""}
    monkeypatch.setattr(validators, "air", lambda source: air["source"])
    # Without the Metal compiler the translation is only validated.
    assert decorations.lost(INVARIANT, "metal", "vertex") == ""
    air["source"] = '!5 = !{i32 0, !"air.position", !"air.invariant"}\n'
    assert decorations.lost(INVARIANT, "metal", "vertex") == ""
    air["source"] = '!5 = !{i32 0, !"air.position"}\n'
    assert decorations.lost(INVARIANT, "metal", "vertex") == (
        "the translation loses Decoration.Invariant"
    )
    # A capability has no Metal attribute, so only the decoration is checked.
    air["source"] = '!7 = !{i32 1, !"air.sample", !"air.perspective"}\n'
    assert decorations.lost(SAMPLE, "metal", "fragment") == ""
    air["source"] = '!7 = !{i32 1, !"air.sampler"}\n'
    assert decorations.lost(SAMPLE, "metal", "fragment") == (
        "the translation loses Decoration.Sample"
    )


def test_a_webgl_translation_is_lowered_as_glsl_es_3_10(monkeypatch):
    calls = []

    def lower(language, source, entry, stage, **options):
        calls.append((language, source.splitlines()[0], stage))
        return "OpMemberDecorate %gl_PerVertex 0 Invariant\n", ""

    monkeypatch.setattr(validators, "can_lower", lambda language: language == "glsl")
    monkeypatch.setattr(validators, "lower", lower)
    webgl = "#version 300 es\ninvariant gl_Position;\nvoid main() {}\n"
    assert decorations.lost(INVARIANT, "webgl", webgl) == ""
    assert calls == [("glsl", "#version 310 es", "vertex")]


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
def test_the_metal_compiler_records_the_invariant_attribute_in_air():
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Out { float4 position [[position, invariant]]; };\n"
        "vertex Out probe() { Out out; out.position = float4(1.0); return out; }\n"
    )
    assert decorations.lost(INVARIANT, "metal", source) == ""
    assert decorations.lost(INVARIANT, "metal", source.replace(", invariant", "")) == (
        "the translation loses Decoration.Invariant"
    )


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_a_glsl_es_3_00_vertex_shader():
    source = (
        "#version 300 es\ninvariant gl_Position;\nin vec4 position;\n"
        "void main() {\n    gl_Position = position + vec4(float(gl_VertexID));\n}\n"
    )
    assert decorations.lost(INVARIANT, "webgl", source) == ""
    assert decorations.lost(
        INVARIANT, "webgl", source.replace("invariant gl_Position;\n", "")
    ) == ("the translation loses Decoration.Invariant")
