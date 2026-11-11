"""Execution modes, extension built-ins, draw parameters and specialization constants a translation must keep."""

import dataclasses

import pytest

from tools.coverage import decorations, lowering, validators
from tools.coverage.model import Feature


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


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
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


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
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
