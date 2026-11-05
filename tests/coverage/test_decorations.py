import dataclasses
import re

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


def test_only_graphics_compute_and_resource_features_are_checked():
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
    # A resource keeps only the shape of its image; its buffer decorations are left to running the probe.
    buffer = Feature(
        "glsl",
        "resource",
        "read-only-buffer",
        None,
        group="StorageClass.StorageBuffer Decoration.NonWritable",
    )
    assert decorations.lost(buffer, "spirv", "") == ""


def _texture(name, group):
    return Feature("glsl", "resource", name, None, group=f"OpTypeImage {group}")


def test_a_texture_translation_must_keep_the_shape_of_its_image():
    cube = _texture("texture-cube", "Dim.Cube")
    array = _texture("texture-cube-array", "Dim.Cube")
    line = "%type_cube_image = OpTypeImage %float Cube 2 {arrayed} 0 1 Unknown\n"
    assert decorations.lost(cube, "spirv", line.format(arrayed=0)) == ""
    assert decorations.lost(array, "spirv", line.format(arrayed=1)) == ""
    # An array texture must stay arrayed, and a single texture must not become one.
    assert decorations.lost(array, "spirv", line.format(arrayed=0)) == (
        "the translation loses Dim.Cube"
    )
    assert decorations.lost(cube, "spirv", line.format(arrayed=1)) == (
        "the translation loses Dim.Cube"
    )
    flat = "%7 = OpTypeImage %float 2D 0 0 0 1 Unknown\n"
    assert decorations.lost(cube, "spirv", flat) == "the translation loses Dim.Cube"
    line1d = "OpCapability Sampled1D\n%5 = OpTypeImage %float 1D 0 0 0 1 Unknown\n"
    texture1d = _texture("texture-1d", "Dim.1D Capability.Sampled1D")
    assert decorations.lost(texture1d, "spirv", line1d) == ""
    assert decorations.lost(texture1d, "spirv", line1d.split("\n", 1)[1]) == (
        "the translation loses Capability.Sampled1D"
    )


def test_a_metal_texture_translation_keeps_the_air_type_of_its_shape(monkeypatch):
    air = {"source": ""}
    monkeypatch.setattr(validators, "air", lambda source: air["source"])
    array = _texture("texture-2d-array", "Dim.2D")
    air["source"] = "%struct._texture_2d_array_t = type opaque\n"
    assert decorations.lost(array, "metal", "kernel") == ""
    air["source"] = "%struct._texture_2d_t = type opaque\n"
    assert decorations.lost(array, "metal", "kernel") == (
        "the translation loses Dim.2D"
    )
    # A capability has no AIR type, so a 1D texture is checked for its shape only.
    texture1d = _texture("texture-1d", "Dim.1D Capability.Sampled1D")
    air["source"] = "%struct._texture_1d_t = type opaque\n"
    assert decorations.lost(texture1d, "metal", "kernel") == ""


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
@pytest.mark.parametrize(
    ("texture", "read", "air"),
    [
        (
            "texturecube<float>",
            "tex.sample(smp, float3(fin[0]), level(0.0))",
            "texture_cube",
        ),
        (
            "texturecube_array<float>",
            "tex.sample(smp, float3(fin[0]), 1, level(0.0))",
            "texture_cube_array",
        ),
        ("texture_buffer<float>", "tex.read(0u)", "texture_buffer_1d"),
    ],
)
def test_the_metal_compiler_names_each_texture_shape_in_air(texture, read, air):
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        f"kernel void probe(device float* fin [[buffer(0)]], {texture} tex [[texture(0)]],\n"
        "                  sampler smp [[sampler(0)]]) {\n"
        f"    fin[0] = {read}.x;\n}}\n"
    )
    assert f"%struct._{air}_t = type opaque" in validators.air(source)


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


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
def test_the_metal_compiler_records_barycentrics_views_and_stencil_in_air():
    header = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct In { float2 uv; };\n"
    )
    bary = _fragment(
        "barycentric-coordinate",
        "BuiltIn.BaryCoordKHR Capability.FragmentBarycentricKHR",
    )
    source = header + (
        "fragment float4 probe(In in [[stage_in]], float3 b [[barycentric_coord]]) "
        "{ return float4(b, in.uv.x); }\n"
    )
    assert decorations.lost(bary, "metal", source) == ""
    assert decorations.lost(
        bary,
        "metal",
        source.replace(", float3 b [[barycentric_coord]]", "").replace(
            "float4(b,", "float4(in.uv, 0.0,"
        ),
    ) == ("the translation loses BuiltIn.BaryCoordKHR")
    view = _fragment("view-index", "BuiltIn.ViewIndex Capability.MultiView")
    source = header + (
        "fragment float4 probe(In in [[stage_in]], uint v [[amplification_id]]) "
        "{ return float4(in.uv, float(v), 1.0); }\n"
    )
    assert decorations.lost(view, "metal", source) == ""
    assert decorations.lost(
        view, "metal", source.replace("[[amplification_id]]", "[[primitive_id]]")
    ) == ("the translation loses BuiltIn.ViewIndex")
    stencil = _fragment(
        "stencil-export", "BuiltIn.FragStencilRefEXT Capability.StencilExportEXT"
    )
    source = header + (
        "struct Out { float4 color [[color(0)]]; uint s [[stencil]]; };\n"
        "fragment Out probe(In in [[stage_in]]) { Out out; out.color = float4(in.uv, 0.0, 1.0); "
        "out.s = 1; return out; }\n"
    )
    assert decorations.lost(stencil, "metal", source) == ""


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_extension_built_ins_with_their_capabilities():
    rate = _fragment(
        "shading-rate", "BuiltIn.ShadingRateKHR Capability.FragmentShadingRateKHR"
    )
    source = (
        "#version 460\n#extension GL_EXT_fragment_shading_rate : require\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n    color = vec4(float(gl_ShadingRateEXT));\n}\n"
    )
    assert decorations.lost(rate, "glsl", source) == ""
    assert decorations.lost(
        rate, "glsl", source.replace("gl_ShadingRateEXT", "gl_FrontFacing")
    ) == (
        "the translation loses BuiltIn.ShadingRateKHR, Capability.FragmentShadingRateKHR"
    )
    stencil = _fragment(
        "stencil-export", "BuiltIn.FragStencilRefEXT Capability.StencilExportEXT"
    )
    source = (
        "#version 460\n#extension GL_ARB_shader_stencil_export : require\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n    color = vec4(1.0);\n    gl_FragStencilRefARB = 1;\n}\n"
    )
    assert decorations.lost(stencil, "glsl", source) == ""
    assert decorations.lost(
        stencil, "glsl", source.replace("    gl_FragStencilRefARB = 1;\n", "")
    ) == (
        "the translation loses BuiltIn.FragStencilRefEXT, Capability.StencilExportEXT"
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


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
def test_the_metal_compiler_records_the_base_vertex_and_instance_in_air():
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "vertex float4 probe(uint base [[base_vertex]]) { return float4(float(base)); }\n"
    )
    base = Feature(
        "metal",
        "graphics",
        "base-vertex",
        None,
        group="BuiltIn.BaseVertex Capability.DrawParameters",
        stage="vertex",
    )
    assert decorations.lost(base, "metal", source) == ""
    assert decorations.lost(
        base, "metal", source.replace("[[base_vertex]]", "[[vertex_id]]")
    ) == ("the translation loses BuiltIn.BaseVertex")


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_a_draw_index_read_with_its_capability():
    source = (
        "#version 460\nvoid main() {\n"
        "    gl_Position = vec4(float(gl_DrawID), 0.0, 0.0, 1.0);\n}\n"
    )
    draw = Feature(
        "glsl",
        "graphics",
        "draw-index",
        None,
        group="BuiltIn.DrawIndex Capability.DrawParameters",
        stage="vertex",
    )
    assert decorations.lost(draw, "glsl", source) == ""
    assert decorations.lost(
        draw, "glsl", source.replace("gl_DrawID", "gl_VertexID")
    ) == ("the translation loses BuiltIn.DrawIndex, Capability.DrawParameters")


_SPECIALIZATION = Feature(
    "wgsl",
    "compute",
    "specialization-constant",
    None,
    group="OpSpecConstant Decoration.SpecId",
)


def test_a_wgsl_translation_keeps_a_specialization_constant_as_an_override():
    # naga resolves overrides before it writes SPIR-V, so the declaration is checked in the translation itself and no
    # compiler is needed.
    source = (
        "override scale: i32 = 3;\n"
        "@group(0) @binding(0) var<storage, read_write> outp: array<f32>;\n"
        "@compute @workgroup_size(1, 1, 1)\n"
        "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        "    outp[id.x] = f32(scale);\n}\n"
    )
    assert decorations.lost(_SPECIALIZATION, "wgsl", source) == ""
    assert decorations.lost(
        _SPECIALIZATION, "wgsl", source.replace("override scale", "const scale")
    ) == ("the translation loses Decoration.SpecId")


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
def test_the_metal_compiler_lists_function_constants_in_air():
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "constant int scale [[function_constant(0)]];\n"
        "kernel void probe(device float* outp [[buffer(0)]]) { outp[0] = float(scale); }\n"
    )
    metal = dataclasses.replace(_SPECIALIZATION, language="metal")
    assert decorations.lost(metal, "metal", source) == ""
    assert decorations.lost(
        metal, "metal", source.replace(" [[function_constant(0)]]", " = 3")
    ) == ("the translation loses Decoration.SpecId")


@pytest.mark.skipif(not validators.command("metal"), reason="Metal is not installed")
@pytest.mark.parametrize(
    ("name", "anchor", "qualified"),
    [
        ("restrict-buffer", "Decoration.Restrict", "device float* __restrict outp"),
        ("volatile-buffer", "Decoration.Volatile", "volatile device float* outp"),
        (
            "coherent-buffer",
            "Decoration.Coherent",
            "coherent(device) device float* outp",
        ),
    ],
)
def test_the_metal_compiler_keeps_buffer_memory_qualifiers_in_air(
    name, anchor, qualified
):
    # AIR has no attribute for these qualifiers: a restrict buffer is a noalias argument, a volatile one is stored to
    # with volatile stores, and a coherent one through the device_coherent intrinsics.
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        f"kernel void probe(device float* fin [[buffer(0)]], {qualified} [[buffer(1)]],\n"
        "                  uint3 id [[thread_position_in_grid]]) {\n"
        "    outp[id.x] = fin[id.x] * 2.0;\n}\n"
    )
    feature = Feature(
        "metal", "compute", name, None, group=f"StorageClass.StorageBuffer {anchor}"
    )
    assert decorations.lost(feature, "metal", source) == ""
    assert decorations.lost(
        feature, "metal", source.replace(qualified, "device float* outp")
    ) == (f"the translation loses {anchor}")


def _geometry(language, name, group):
    return Feature(language, "graphics", name, None, group=group, stage="geometry")


def test_an_instanced_geometry_translation_must_keep_more_than_one_invocation():
    # glslang, DXC and slangc write Invocations 1 for every geometry stage, so only a larger count keeps the mode.
    instanced = _geometry(
        "spirv",
        "geometry-invocations",
        "ExecutionMode.Invocations BuiltIn.InvocationId",
    )
    kept = (
        "OpExecutionMode %main InputPoints\nOpExecutionMode %main Invocations 2\n"
        "OpDecorate %gl_InvocationID BuiltIn InvocationId\n"
    )
    assert decorations.lost(instanced, "spirv", kept) == ""
    assert (
        decorations.lost(
            instanced, "spirv", kept.replace("Invocations 2", "Invocations 12")
        )
        == ""
    )
    assert decorations.lost(
        instanced, "spirv", kept.replace("Invocations 2", "Invocations 1")
    ) == ("the translation loses ExecutionMode.Invocations")
    lines = _geometry("spirv", "geometry-input-lines", "ExecutionMode.InputLines")
    assert decorations.lost(lines, "spirv", "OpExecutionMode %main InputLines\n") == ""
    assert decorations.lost(lines, "spirv", "OpExecutionMode %main InputPoints\n") == (
        "the translation loses ExecutionMode.InputLines"
    )


@pytest.mark.parametrize("language", ["hlsl", "slang"])
def test_an_hlsl_or_slang_stream_translation_keeps_a_second_stream_parameter(language):
    # DXC writes every HLSL output on stream 0 in SPIR-V and slangc cannot write a second stream there, so the
    # translation itself must declare a second point stream.
    streams = _geometry(
        language, "geometry-streams", "Decoration.Stream Capability.GeometryStreams"
    )
    source = (
        "[maxvertexcount(2)]\n"
        "void main(point float4 position[1] : POSITION, inout PointStream<GeometryOut> stream,\n"
        "          inout PointStream<ExtraOut> extras) {\n}\n"
    )
    assert decorations.lost(streams, language, source) == ""
    single = source.replace(",\n          inout PointStream<ExtraOut> extras", "")
    assert decorations.lost(streams, language, single) == (
        "the translation loses Decoration.Stream, Capability.GeometryStreams"
    )


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_geometry_primitives_instances_and_streams():
    lines = _geometry("glsl", "geometry-input-lines", "ExecutionMode.InputLines")
    source = (
        "#version 460\nlayout(lines) in;\nlayout(points, max_vertices = 1) out;\n"
        "layout(location = 0) in vec4 position[];\n"
        "void main() {\n    gl_Position = position[0];\n    EmitVertex();\n}\n"
    )
    assert decorations.lost(lines, "glsl", source) == ""
    assert decorations.lost(lines, "glsl", source.replace("(lines)", "(points)")) == (
        "the translation loses ExecutionMode.InputLines"
    )
    instanced = _geometry(
        "glsl",
        "geometry-invocations",
        "ExecutionMode.Invocations BuiltIn.InvocationId",
    )
    source = (
        "#version 460\nlayout(points, invocations = 2) in;\nlayout(points, max_vertices = 1) out;\n"
        "void main() {\n    gl_Position = vec4(float(gl_InvocationID));\n    EmitVertex();\n}\n"
    )
    assert decorations.lost(instanced, "glsl", source) == ""
    assert decorations.lost(
        instanced, "glsl", source.replace(", invocations = 2", "")
    ) == ("the translation loses ExecutionMode.Invocations")
    streams = _geometry(
        "glsl", "geometry-streams", "Decoration.Stream Capability.GeometryStreams"
    )
    source = (
        "#version 460\nlayout(points) in;\nlayout(points, max_vertices = 2) out;\n"
        "layout(stream = 1, location = 0) out vec4 extra;\n"
        "void main() {\n    gl_Position = vec4(1.0);\n    EmitStreamVertex(0);\n"
        "    extra = vec4(2.0);\n    EmitStreamVertex(1);\n}\n"
    )
    assert decorations.lost(streams, "glsl", source) == ""
    single = (
        "#version 460\nlayout(points) in;\nlayout(points, max_vertices = 2) out;\n"
        "layout(location = 0) out vec4 extra;\n"
        "void main() {\n    gl_Position = vec4(1.0);\n    extra = vec4(2.0);\n    EmitVertex();\n}\n"
    )
    assert decorations.lost(streams, "glsl", single) == (
        "the translation loses Decoration.Stream, Capability.GeometryStreams"
    )


def _patch(language, name, group, stage="tessellation-control"):
    return Feature(language, "graphics", name, None, group=group, stage=stage)


def test_a_per_patch_translation_must_keep_a_located_patch_variable():
    # Every tessellation stage decorates its levels Patch, so only a Patch variable with a Location keeps the value.
    output = _patch("spirv", "tessellation-patch-output", "Decoration.Patch")
    levels = "OpDecorate %gl_TessLevelOuter BuiltIn TessLevelOuter\nOpDecorate %gl_TessLevelOuter Patch\n"
    assert decorations.lost(output, "spirv", levels) == (
        "the translation loses Decoration.Patch"
    )
    assert decorations.lost(
        output, "spirv", levels + "OpDecorate %20 Location 1\nOpDecorate %24 Patch\n"
    ) == ("the translation loses Decoration.Patch")
    for kept in (
        "OpDecorate %extra Patch\nOpDecorate %extra Location 1\n",
        "OpDecorate %extra Location 1\nOpDecorate %extra Patch\n",
    ):
        assert decorations.lost(output, "spirv", levels + kept) == ""


def test_a_slang_per_patch_translation_keeps_a_user_semantic_beside_its_levels():
    # slangc writes no Patch decoration for a patch-constant member, so the translation itself must declare one.
    output = _patch("slang", "tessellation-patch-output", "Decoration.Patch")
    source = (
        "struct PatchConstants {\n    float edges[3] : SV_TessFactor;\n"
        "    float inside : SV_InsideTessFactor;\n    float4 extra : EXTRA;\n};\n"
    )
    assert decorations.lost(output, "slang", source) == ""
    assert decorations.lost(
        output, "slang", source.replace("    float4 extra : EXTRA;\n", "")
    ) == ("the translation loses Decoration.Patch")
    unrelated = "struct Other {\n    float4 extra : EXTRA;\n};\n"
    assert decorations.lost(
        output,
        "slang",
        unrelated + source.replace("    float4 extra : EXTRA;\n", ""),
    ) == ("the translation loses Decoration.Patch")


def test_a_metal_tessellation_translation_keeps_its_quad_patch_and_per_patch_input():
    quads = _patch(
        "metal",
        "tessellation-domain-quads",
        "ExecutionMode.Quads",
        "tessellation-evaluation",
    )
    patch_input = _patch(
        "metal",
        "tessellation-patch-input",
        "Decoration.Patch",
        "tessellation-evaluation",
    )
    assert decorations._wanted(quads, "metal") == [
        ("ExecutionMode.Quads", r'!"air\.patch", !"quad"')
    ]
    assert decorations._wanted(patch_input, "metal") == [
        ("Decoration.Patch", r'!"air\.patch_input"')
    ]


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_tessellation_domains_spacing_and_patch_values():
    quads = _patch(
        "glsl",
        "tessellation-domain-quads",
        "ExecutionMode.Quads",
        "tessellation-evaluation",
    )
    source = (
        "#version 460\nlayout(quads, equal_spacing, ccw) in;\nlayout(location = 0) in vec4 position[];\n"
        "void main() {\n    gl_Position = position[0] * gl_TessCoord.x;\n}\n"
    )
    assert decorations.lost(quads, "glsl", source) == ""
    assert decorations.lost(quads, "glsl", source.replace("quads", "triangles")) == (
        "the translation loses ExecutionMode.Quads"
    )
    odd = _patch(
        "glsl",
        "tessellation-spacing-fractional-odd",
        "ExecutionMode.SpacingFractionalOdd",
        "tessellation-evaluation",
    )
    spaced = source.replace("equal_spacing", "fractional_odd_spacing")
    assert decorations.lost(odd, "glsl", spaced) == ""
    assert decorations.lost(odd, "glsl", source) == (
        "the translation loses ExecutionMode.SpacingFractionalOdd"
    )
    output = _patch("glsl", "tessellation-patch-output", "Decoration.Patch")
    source = (
        "#version 460\nlayout(vertices = 3) out;\nlayout(location = 0) in vec4 position[];\n"
        "layout(location = 0) out vec4 outPosition[];\nlayout(location = 1) patch out vec4 extra;\n"
        "void main() {\n    outPosition[gl_InvocationID] = position[gl_InvocationID];\n"
        "    extra = position[0];\n    gl_TessLevelOuter[0] = 1.0;\n}\n"
    )
    assert decorations.lost(output, "glsl", source) == ""
    dropped = source.replace(
        "layout(location = 1) patch out vec4 extra;\n", ""
    ).replace("    extra = position[0];\n", "")
    assert decorations.lost(output, "glsl", dropped) == (
        "the translation loses Decoration.Patch"
    )


def _mesh(language, name, group):
    return Feature(language, "graphics", name, None, group=group, stage="mesh")


def test_a_mesh_translation_must_keep_its_topology_and_per_primitive_values():
    lines = _mesh(
        "spirv",
        "mesh-output-lines",
        "ExecutionMode.OutputLinesEXT BuiltIn.PrimitiveLineIndicesEXT",
    )
    kept = (
        "OpExecutionMode %main OutputLinesEXT\n"
        "OpDecorate %gl_PrimitiveLineIndicesEXT BuiltIn PrimitiveLineIndicesEXT\n"
    )
    assert decorations.lost(lines, "spirv", kept) == ""
    # A disassembler built from a grammar older than SPV_EXT_mesh_shader names the shared enumerants by their NV
    # spelling.
    assert (
        decorations.lost(lines, "spirv", kept.replace("LinesEXT\n", "LinesNV\n")) == ""
    )
    triangles = (
        "OpExecutionMode %main OutputTrianglesEXT\n"
        "OpDecorate %gl_PrimitiveTriangleIndicesEXT BuiltIn PrimitiveTriangleIndicesEXT\n"
    )
    assert decorations.lost(lines, "spirv", triangles) == (
        "the translation loses ExecutionMode.OutputLinesEXT, BuiltIn.PrimitiveLineIndicesEXT"
    )
    output = _mesh("spirv", "mesh-per-primitive-output", "Decoration.PerPrimitiveEXT")
    for spelling in ("PerPrimitiveEXT", "PerPrimitiveNV"):
        assert (
            decorations.lost(output, "spirv", f"OpDecorate %color {spelling}\n") == ""
        )
    assert decorations.lost(output, "spirv", "OpDecorate %color Location 0\n") == (
        "the translation loses Decoration.PerPrimitiveEXT"
    )


def test_a_mesh_translation_that_reads_the_payload_must_declare_task_payload_storage():
    # A storage class is no decoration, so the payload is kept only by a variable in task-payload storage.
    payload = _mesh(
        "spirv", "mesh-task-payload", "StorageClass.TaskPayloadWorkgroupEXT"
    )
    pointer = "%_ptr_TaskPayloadWorkgroupEXT_Payload = OpTypePointer TaskPayloadWorkgroupEXT %Payload\n"
    variable = "%payload = OpVariable %_ptr_TaskPayloadWorkgroupEXT_Payload TaskPayloadWorkgroupEXT\n"
    assert decorations.lost(payload, "spirv", pointer + variable) == ""
    assert decorations.lost(payload, "spirv", pointer) == (
        "the translation loses StorageClass.TaskPayloadWorkgroupEXT"
    )
    private = variable.replace(" TaskPayloadWorkgroupEXT\n", " Private\n")
    assert decorations.lost(payload, "spirv", pointer + private) == (
        "the translation loses StorageClass.TaskPayloadWorkgroupEXT"
    )


def test_a_metal_mesh_translation_keeps_its_topology_primitive_data_and_payload_in_air():
    points = _mesh(
        "metal",
        "mesh-output-points",
        "ExecutionMode.OutputPoints BuiltIn.PrimitivePointIndicesEXT",
    )
    ((anchor, pattern),) = decorations._wanted(points, "metal")
    assert anchor == "ExecutionMode.OutputPoints"
    info = '!12 = !{!"air.mesh_type_info", !13, !14, i32 1, i32 1, !"air.point"}\n'
    assert re.search(pattern, info, re.M)
    assert not re.search(pattern, info.replace("air.point", "air.triangle"), re.M)
    for name, group, attribute in (
        (
            "mesh-per-primitive-output",
            "Decoration.PerPrimitiveEXT",
            "mesh_primitive_data",
        ),
        ("mesh-cull-primitive", "BuiltIn.CullPrimitiveEXT", "primitive_culled"),
        ("mesh-task-payload", "StorageClass.TaskPayloadWorkgroupEXT", "payload"),
    ):
        assert decorations._wanted(_mesh("metal", name, group), "metal") == [
            (group, rf'!"air\.{attribute}"')
        ]


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_mesh_topologies_per_primitive_outputs_and_payloads():
    header = "#version 460\n#extension GL_EXT_mesh_shader : require\nlayout(local_size_x = 1) in;\n"
    lines = _mesh(
        "glsl",
        "mesh-output-lines",
        "ExecutionMode.OutputLinesEXT BuiltIn.PrimitiveLineIndicesEXT",
    )
    source = (
        header + "layout(lines, max_vertices = 2, max_primitives = 1) out;\n"
        "void main() {\n    SetMeshOutputsEXT(2, 1);\n"
        "    gl_PrimitiveLineIndicesEXT[0] = uvec2(0, 1);\n}\n"
    )
    assert decorations.lost(lines, "glsl", source) == ""
    triangles = (
        header + "layout(triangles, max_vertices = 3, max_primitives = 1) out;\n"
        "void main() {\n    SetMeshOutputsEXT(3, 1);\n"
        "    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);\n}\n"
    )
    assert decorations.lost(lines, "glsl", triangles) == (
        "the translation loses ExecutionMode.OutputLinesEXT, BuiltIn.PrimitiveLineIndicesEXT"
    )
    output = _mesh("glsl", "mesh-per-primitive-output", "Decoration.PerPrimitiveEXT")
    located = triangles.replace(
        "void main() {\n",
        "layout(location = 0) perprimitiveEXT out vec4 color[];\nvoid main() {\n",
    ).replace("}\n", "    color[0] = vec4(1.0);\n}\n")
    assert decorations.lost(output, "glsl", located) == ""
    assert decorations.lost(
        output, "glsl", located.replace("perprimitiveEXT ", "")
    ) == ("the translation loses Decoration.PerPrimitiveEXT")
    payload = _mesh("glsl", "mesh-task-payload", "StorageClass.TaskPayloadWorkgroupEXT")
    reads = triangles.replace(
        "void main() {\n",
        "struct Payload {\n    uint value;\n};\ntaskPayloadSharedEXT Payload payload;\n"
        "void main() {\n",
    ).replace(
        "}\n",
        "    gl_MeshVerticesEXT[0].gl_Position = vec4(float(payload.value));\n}\n",
    )
    assert decorations.lost(payload, "glsl", reads) == ""
    assert decorations.lost(payload, "glsl", triangles) == (
        "the translation loses StorageClass.TaskPayloadWorkgroupEXT"
    )


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_the_ids_a_task_stage_reads():
    feature = Feature(
        "glsl",
        "graphics",
        "task-workgroup-id",
        None,
        group="BuiltIn.WorkgroupId",
        stage="task",
    )
    source = (
        "#version 460\n#extension GL_EXT_mesh_shader : require\nlayout(local_size_x = 1) in;\n"
        "struct Payload {\n    uint value;\n};\ntaskPayloadSharedEXT Payload payload;\n"
        "void main() {\n    payload.value = gl_WorkGroupID.x;\n    EmitMeshTasksEXT(1, 1, 1);\n}\n"
    )
    assert decorations.lost(feature, "glsl", source) == ""
    assert decorations.lost(
        feature, "glsl", source.replace("gl_WorkGroupID.x", "0u")
    ) == ("the translation loses BuiltIn.WorkgroupId")


def _ray(name, group, stage="ray-generation"):
    return Feature("spirv", "graphics", name, None, group=group, stage=stage)


def test_a_ray_tracing_translation_keeps_the_storage_and_calls_that_hand_data_over():
    # A storage class is no decoration, so a payload is kept only by a variable in ray-payload storage, and is
    # handed over only by the trace that takes it.
    payload = _ray("ray-payload", "StorageClass.RayPayloadKHR OpTraceRayKHR")
    variable = "%payload = OpVariable %_ptr_RayPayloadKHR_Payload RayPayloadKHR\n"
    trace = "OpTraceRayKHR %scene %uint_0 %uint_255 %uint_0 %uint_1 %uint_0 %o %t %d %m %payload\n"
    assert decorations.lost(payload, "spirv", variable + trace) == ""
    # SPV_NV_ray_tracing gave the storage classes their values first, so an older disassembler prints NV names.
    nv = variable.replace("RayPayloadKHR", "RayPayloadNV")
    assert decorations.lost(payload, "spirv", nv + trace) == ""
    assert decorations.lost(payload, "spirv", variable) == (
        "the translation loses OpTraceRayKHR"
    )
    private = variable.replace(" RayPayloadKHR\n", " Private\n")
    assert decorations.lost(payload, "spirv", private + trace) == (
        "the translation loses StorageClass.RayPayloadKHR"
    )
    record = _ray("ray-shader-record", "StorageClass.ShaderRecordBufferKHR", "miss")
    assert (
        decorations.lost(
            record, "spirv", "%record = OpVariable %_ptr_Record ShaderRecordBufferNV\n"
        )
        == ""
    )
    assert decorations.lost(
        record, "spirv", "%record = OpVariable %_ptr_Record StorageBuffer\n"
    ) == ("the translation loses StorageClass.ShaderRecordBufferKHR")
    reported = _ray(
        "ray-reported-attributes",
        "StorageClass.HitAttributeKHR OpReportIntersectionKHR",
        "intersection",
    )
    attributes = (
        "%Attributes = OpTypeStruct %v3float\n"
        "%_ptr_HitAttributeKHR_Attributes = OpTypePointer HitAttributeKHR %Attributes\n"
    )
    report = "%hit = OpReportIntersectionNV %bool %float_1 %uint_0\n"
    assert decorations.lost(reported, "spirv", attributes + report) == ""
    # The barycentrics a triangle hit carries are a vector, not attributes of the stage's own.
    barycentrics = (
        "%_ptr_HitAttributeKHR_v2float = OpTypePointer HitAttributeKHR %v2float\n"
    )
    assert decorations.lost(reported, "spirv", barycentrics + report) == (
        "the translation loses StorageClass.HitAttributeKHR"
    )


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_the_data_ray_tracing_stages_hand_over():
    callable_data = Feature(
        "glsl",
        "graphics",
        "ray-callable-data",
        None,
        group="StorageClass.CallableDataKHR OpExecuteCallableKHR",
        stage="ray-generation",
    )
    source = (
        "#version 460\n#extension GL_EXT_ray_tracing : require\n"
        "struct CallData {\n    vec4 color;\n};\nlayout(location = 0) callableDataEXT CallData data;\n"
        "layout(std430, binding = 0) buffer Output {\n    float outp[];\n};\n"
        "void main() {\n    data.color = vec4(0.0);\n    executeCallableEXT(0u, 0);\n"
        "    outp[0] = data.color.x;\n}\n"
    )
    assert decorations.lost(callable_data, "glsl", source) == ""
    assert decorations.lost(
        callable_data, "glsl", source.replace("    executeCallableEXT(0u, 0);\n", "")
    ) == ("the translation loses OpExecuteCallableKHR")
    attributes = Feature(
        "glsl",
        "graphics",
        "ray-custom-attributes",
        None,
        group="StorageClass.HitAttributeKHR OpTypeStruct",
        stage="closest-hit",
    )
    source = (
        "#version 460\n#extension GL_EXT_ray_tracing : require\n"
        "struct Payload {\n    vec4 color;\n};\nlayout(location = 0) rayPayloadInEXT Payload payload;\n"
        "struct Attributes {\n    vec3 normal;\n};\nhitAttributeEXT Attributes attributes;\n"
        "void main() {\n    payload.color = vec4(attributes.normal, 1.0);\n}\n"
    )
    assert decorations.lost(attributes, "glsl", source) == ""
    barycentrics = source.replace(
        "hitAttributeEXT Attributes attributes;",
        "hitAttributeEXT vec2 uv;\nAttributes attributes;",
    )
    assert decorations.lost(attributes, "glsl", barycentrics) == (
        "the translation loses StorageClass.HitAttributeKHR"
    )


def _resource(name, group):
    return Feature("spirv", "resource", name, None, group=group)


def test_a_binding_translation_keeps_its_push_constants_and_descriptor_set():
    # A storage class is no decoration, so push constants are kept only by a variable in push-constant storage.
    push = _resource("push-constant", "StorageClass.PushConstant")
    variable = "%pc = OpVariable %_ptr_PushConstant_Constants PushConstant\n"
    assert decorations.lost(push, "spirv", variable) == ""
    uniform = "%pc = OpVariable %_ptr_Uniform_Constants Uniform\n"
    assert decorations.lost(push, "spirv", uniform) == (
        "the translation loses StorageClass.PushConstant"
    )
    # Every buffer is decorated with its set, so only a set other than the first keeps the probe's.
    bound = _resource("descriptor-set", "Decoration.DescriptorSet")
    first = "OpDecorate %fin DescriptorSet 0\nOpDecorate %params DescriptorSet 0\n"
    assert decorations.lost(bound, "spirv", first) == (
        "the translation loses Decoration.DescriptorSet"
    )
    second = first.replace("%params DescriptorSet 0", "%params DescriptorSet 1")
    assert decorations.lost(bound, "spirv", second) == ""
    assert decorations.lost(bound, "spirv", second.replace("Set 1", "Set 10")) == ""


@pytest.mark.skipif(not validators.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_push_constants_and_a_second_descriptor_set():
    head = (
        "#version 460\nlayout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        "layout(std430, binding = 0) buffer B0 { float fin[]; };\n"
        "layout(std430, binding = 1) buffer B1 { float outp[]; };\n"
    )
    body = "void main() {\n    outp[0] = fin[0] * scale;\n}\n"
    push = Feature(
        "glsl", "resource", "push-constant", None, group="StorageClass.PushConstant"
    )
    pushed = (
        head
        + "layout(push_constant) uniform Constants {\n    float scale;\n};\n"
        + body
    )
    assert decorations.lost(push, "glsl", pushed) == ""
    uniform = pushed.replace("push_constant", "std140, binding = 3")
    assert decorations.lost(push, "glsl", uniform) == (
        "the translation loses StorageClass.PushConstant"
    )
    bound = Feature(
        "glsl", "resource", "descriptor-set", None, group="Decoration.DescriptorSet"
    )
    assert decorations.lost(bound, "glsl", uniform) == (
        "the translation loses Decoration.DescriptorSet"
    )
    second = uniform.replace("std140, binding = 3", "std140, set = 1, binding = 3")
    assert decorations.lost(bound, "glsl", second) == ""
