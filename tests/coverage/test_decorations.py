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


def test_only_graphics_and_compute_features_are_checked():
    loop = Feature("glsl", "control-flow", "for-loop", None, group="OpLoopMerge")
    shift = Feature("glsl", "operator", "vector-shift", None, group="Decoration.Flat")
    assert decorations.lost(loop, "spirv", "") == ""
    assert decorations.lost(shift, "spirv", "") == ""
    # Workgroup memory and barriers are instructions and storage classes, which only running the probe checks.
    shared = Feature(
        "glsl",
        "compute",
        "shared-memory",
        None,
        group="StorageClass.Workgroup OpControlBarrier",
    )
    assert decorations.lost(shared, "spirv", "") == ""


def _compute(name, group):
    return Feature("glsl", "compute", name, None, group=group)


def test_a_compute_translation_must_keep_its_built_in():
    # One invocation per workgroup makes the lane index 0, so only the decoration tells it from a constant.
    lane = _compute("subgroup-invocation-id", "BuiltIn.SubgroupLocalInvocationId")
    decorated = (
        "OpDecorate %gl_SubgroupInvocationID BuiltIn SubgroupLocalInvocationId\n"
    )
    assert decorations.lost(lane, "spirv", decorated) == ""
    assert decorations.lost(lane, "spirv", "%float_0 = OpConstant %float 0\n") == (
        "the translation loses BuiltIn.SubgroupLocalInvocationId"
    )


def test_a_constant_built_in_may_be_written_as_its_value(monkeypatch):
    size = _compute("workgroup-size", "BuiltIn.WorkgroupSize")
    monkeypatch.setattr(validators, "can_lower", lambda language: True)
    monkeypatch.setattr(
        validators,
        "lower",
        lambda *args, **kwargs: ("%uint_1 = OpConstant %uint 1\n", ""),
    )
    assert decorations.lost(size, "slang", "outp[i] = float(WorkgroupSize().x);") == ""
    assert decorations.lost(size, "spirv", "%uint_1 = OpConstant %uint 1\n") == ""


def test_a_language_without_the_feature_only_validates_its_translation():
    # HLSL has no workgroup count, so the HLSL that CrossTL writes for an OpenCL C probe cannot keep one.
    count = _compute("num-workgroups", "BuiltIn.NumWorkgroups")
    assert decorations.lost(count, "hlsl", "outp[i] = float(_num_groups.x);") == ""
    assert decorations.lost(count, "spirv", "%uint_1 = OpConstant %uint 1\n") == (
        "the translation loses BuiltIn.NumWorkgroups"
    )


def test_a_metal_compute_translation_keeps_the_air_of_its_built_in(monkeypatch):
    monkeypatch.setattr(validators, "air", lambda source: source)
    groups = _compute("num-subgroups", "BuiltIn.NumSubgroups")
    assert (
        decorations.lost(groups, "metal", '!{!"air.simdgroups_per_threadgroup"}') == ""
    )
    assert decorations.lost(groups, "metal", '!{!"air.thread_position_in_grid"}') == (
        "the translation loses BuiltIn.NumSubgroups"
    )


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
def test_the_metal_compiler_records_the_subgroup_attributes_in_air():
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void probe(device float* outp [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {\n"
        "    outp[0] = float(lane);\n}\n"
    )
    lane = _compute("subgroup-invocation-id", "BuiltIn.SubgroupLocalInvocationId")
    assert decorations.lost(lane, "metal", source) == ""
    assert decorations.lost(
        lane,
        "metal",
        source.replace(
            "lane [[thread_index_in_simdgroup]]", "lane [[thread_index_in_threadgroup]]"
        ),
    ) == ("the translation loses BuiltIn.SubgroupLocalInvocationId")


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


def test_a_metal_coverage_mask_input_has_its_own_air_attribute(monkeypatch):
    # AIR names the coverage mask a fragment function reads apart from the one it writes.
    mask = {
        name: Feature(
            "glsl", "graphics", name, None, group="BuiltIn.SampleMask", stage="fragment"
        )
        for name in ("sample-mask-input", "sample-mask")
    }
    monkeypatch.setattr(validators, "air", lambda source: source)
    read = '!9 = !{i32 1, !"air.sample_mask_in"}\n'
    written = '!9 = !{!"air.sample_mask"}\n'
    assert decorations.lost(mask["sample-mask-input"], "metal", read) == ""
    assert decorations.lost(mask["sample-mask"], "metal", written) == ""
    assert decorations.lost(mask["sample-mask-input"], "metal", written) == (
        "the translation loses BuiltIn.SampleMask"
    )


def _fragment(name, group):
    return Feature("hlsl", "graphics", name, None, group=group, stage="fragment")


def test_a_translation_must_keep_each_execution_mode_of_its_feature():
    depth = _fragment(
        "conservative-depth", "BuiltIn.FragDepth ExecutionMode.DepthGreater"
    )
    kept = (
        "OpExecutionMode %main OriginUpperLeft\nOpExecutionMode %main DepthReplacing\n"
        "OpExecutionMode %main DepthGreater\nOpDecorate %depth BuiltIn FragDepth\n"
    )
    assert decorations.lost(depth, "spirv", kept) == ""
    assert decorations.lost(
        depth, "spirv", kept.replace("DepthGreater", "DepthGreaterX")
    ) == ("the translation loses ExecutionMode.DepthGreater")
    early = _fragment("early-fragment-tests", "ExecutionMode.EarlyFragmentTests")
    assert decorations.lost(
        early, "spirv", "OpExecutionMode %main OriginUpperLeft\n"
    ) == ("the translation loses ExecutionMode.EarlyFragmentTests")


def test_a_helper_query_may_be_the_built_in_or_the_demote_instruction():
    # DXC and slangc lower IsHelperLane() to OpIsHelperInvocationEXT; glslang reads the built-in.
    helper = _fragment("helper-invocation", "BuiltIn.HelperInvocation")
    builtin = "OpDecorate %gl_HelperInvocation BuiltIn HelperInvocation\n"
    instruction = "         %12 = OpIsHelperInvocationEXT %bool\n"
    assert decorations.lost(helper, "spirv", builtin) == ""
    assert decorations.lost(helper, "spirv", instruction) == ""
    assert decorations.lost(helper, "spirv", "%12 = OpConstantTrue %bool\n") == (
        "the translation loses BuiltIn.HelperInvocation"
    )


def test_a_metal_translation_keeps_the_air_of_each_pipeline_feature(monkeypatch):
    monkeypatch.setattr(validators, "air", lambda source: source)
    helper = _fragment("helper-invocation", "BuiltIn.HelperInvocation")
    call = "%3 = tail call i1 @air.simd_is_helper_thread() #2\n"
    assert decorations.lost(helper, "metal", call) == ""
    assert decorations.lost(helper, "metal", "%3 = icmp eq i32 0, 0\n") == (
        "the translation loses BuiltIn.HelperInvocation"
    )
    # The Metal compiler writes the early tests without the air prefix.
    early = _fragment("early-fragment-tests", "ExecutionMode.EarlyFragmentTests")
    assert (
        decorations.lost(
            early, "metal", '!9 = !{ptr @probe, !"early_fragment_tests"}\n'
        )
        == ""
    )
    depth = _fragment(
        "conservative-depth", "BuiltIn.FragDepth ExecutionMode.DepthGreater"
    )
    greater = '!10 = !{!"air.depth", !"air.depth_qualifier", !"air.greater"}\n'
    assert decorations.lost(depth, "metal", greater) == ""
    assert decorations.lost(depth, "metal", greater.replace("greater", "any")) == (
        "the translation loses ExecutionMode.DepthGreater"
    )
    layer = _fragment("layer", "BuiltIn.Layer")
    assert decorations.lost(layer, "metal", '!{!"air.render_target_array_index"}') == ""


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
def test_the_metal_compiler_records_early_tests_and_conservative_depth_in_air():
    header = "#include <metal_stdlib>\nusing namespace metal;\n"
    early = _fragment("early-fragment-tests", "ExecutionMode.EarlyFragmentTests")
    source = (
        header
        + "[[early_fragment_tests]] fragment float4 probe() { return float4(1.0); }\n"
    )
    assert decorations.lost(early, "metal", source) == ""
    assert decorations.lost(
        early, "metal", source.replace("[[early_fragment_tests]] ", "")
    ) == ("the translation loses ExecutionMode.EarlyFragmentTests")
    depth = _fragment(
        "conservative-depth", "BuiltIn.FragDepth ExecutionMode.DepthGreater"
    )
    source = header + (
        "struct Out { float4 color [[color(0)]]; float depth [[depth(greater)]]; };\n"
        "fragment Out probe() { Out out; out.color = float4(1.0); out.depth = 0.5; return out; }\n"
    )
    assert decorations.lost(depth, "metal", source) == ""
    assert decorations.lost(depth, "metal", source.replace("greater", "any")) == (
        "the translation loses ExecutionMode.DepthGreater"
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
