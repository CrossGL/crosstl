"""Probes for vertex and fragment built-ins, interfaces, interpolation and pipeline modes."""

from tools.coverage import constructs, probes
from tools.coverage.templates import (
    draws,
    extensions,
    fragments,
    interfaces,
    interlocks,
    pipeline,
    rasterization,
    vertices,
)

from .construct_grammar import GRAMMAR, construct_feature


def test_rasterizer_ordered_probes_order_the_accesses_of_overlapping_fragments():
    feature = construct_feature("glsl", "rasterizer-ordered-image")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    assert feature.group == (
        "ExecutionMode.PixelInterlockOrderedEXT Capability.FragmentShaderPixelInterlockEXT"
        " OpBeginInvocationInterlockEXT"
    )
    for language, name, line in (
        ("glsl", "rasterizer-ordered-image", "layout(pixel_interlock_ordered) in;\n"),
        ("glsl", "rasterizer-ordered-buffer", "    beginInvocationInterlockARB();\n"),
        ("glsl", "rasterizer-ordered-image", "    endInvocationInterlockARB();\n"),
        (
            "hlsl",
            "rasterizer-ordered-image",
            "RasterizerOrderedTexture2D<float4> accum : register(u1);\n",
        ),
        (
            "slang",
            "rasterizer-ordered-buffer",
            "RasterizerOrderedStructuredBuffer<float4> accum : register(u1);\n",
        ),
        ("metal", "rasterizer-ordered-image", "[[texture(0), raster_order_group(0)]]"),
        ("metal", "rasterizer-ordered-buffer", "[[buffer(0), raster_order_group(0)]]"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    for name, line in (
        (
            "rasterizer-ordered-image",
            "    image2D accum @rasterizer_ordered @rgba32f;\n",
        ),
        (
            "rasterizer-ordered-buffer",
            "    RWStructuredBuffer<vec4> accum @rasterizer_ordered;\n",
        ),
    ):
        assert (
            line in probes.build(construct_feature("glsl", name), "crossgl").source
        ), name
    # WGSL and GLSL ES 3.00 have no ordered fragment access, the kernel languages no fragment stage, and rust-gpu
    # cannot open the interlocked region.
    names = set(interlocks.NAMES)
    for language in ("glsl", "hlsl", "slang", "metal"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert names <= listed, language
    for language in ("wgsl", "webgl", "rust", "cuda", "hip", "opencl", "mojo"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not listed & names, language


def test_fragment_probes_use_each_languages_built_ins():
    # The built-ins are graphics features of the fragment stage, which the kernel languages do not have.
    feature = construct_feature("glsl", "front-facing")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    for language, name, line in (
        (
            "glsl",
            "fragment-coordinate",
            "vec4(gl_FragCoord.xy * 0.5 + uv, gl_FragCoord.z, 1.0)",
        ),
        ("webgl", "fragment-depth", "gl_FragDepth = uv.x * 0.5 + 0.25;"),
        (
            "hlsl",
            "front-facing",
            "float4 main(float2 uv : TEXCOORD0, bool front : SV_IsFrontFace)",
        ),
        (
            "slang",
            "fragment-depth",
            '[shader("fragment")]\nFragmentOutput main(float2 uv : TEXCOORD0)',
        ),
        ("metal", "fragment-depth", "float depth [[depth(any)]];"),
        ("wgsl", "fragment-coordinate", "@builtin(position) position: vec4<f32>"),
        ("rust", "fragment-depth", "#[spirv(fragment(depth_replacing))]"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, language
    assert (
        probes.build(construct_feature("glsl", "fragment-depth"), "glsl").suffix
        == ".frag"
    )
    hlsl = probes.build(construct_feature("hlsl", "fragment-depth"), "hlsl").source
    assert hlsl.startswith("struct FragmentOutput {") and "[shader(" not in hlsl
    crossgl = probes.build(
        construct_feature("glsl", "fragment-depth"), "crossgl"
    ).source
    assert "        float depth @ gl_FragDepth;\n" in crossgl
    assert "output.depth = input.uv.x * 0.5 + 0.25;" in crossgl
    crossgl = probes.build(construct_feature("glsl", "front-facing"), "crossgl").source
    assert "bool front @ gl_FrontFacing;" in crossgl and "input.front ? " in crossgl
    for language in ("cuda", "hip", "opencl", "mojo"):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not names & set(fragments.NAMES), language


def test_interface_probes_qualify_the_fragment_input_or_add_a_second_target():
    feature = construct_feature("glsl", "sample-interpolation")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    assert feature.group == "Decoration.Sample Capability.SampleRateShading"
    for language, name, line in (
        (
            "glsl",
            "noperspective-interpolation",
            "layout(location = 0) noperspective in vec2 uv;",
        ),
        ("webgl", "centroid-interpolation", "centroid in vec2 uv;"),
        (
            "hlsl",
            "flat-interpolation",
            "float4 main(nointerpolation float2 uv : TEXCOORD0)",
        ),
        ("slang", "sample-interpolation", "float4 main(sample float2 uv : TEXCOORD0)"),
        (
            "metal",
            "noperspective-interpolation",
            "float2 uv [[center_no_perspective]];",
        ),
        (
            "wgsl",
            "centroid-interpolation",
            "@interpolate(perspective, centroid) uv: vec2<f32>",
        ),
        ("rust", "flat-interpolation", "#[spirv(flat)] uv: Vec2"),
        ("glsl", "multiple-render-targets", "layout(location = 1) out vec4 extra;"),
        ("hlsl", "multiple-render-targets", "float4 extra : SV_Target1;"),
        ("metal", "multiple-render-targets", "float4 extra [[color(1)]];"),
        ("rust", "multiple-render-targets", "#[spirv(location = 1)] extra: &mut Vec4"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "flat-interpolation"), "crossgl"
    ).source
    assert "        vec2 uv @ flat;\n" in crossgl
    crossgl = probes.build(
        construct_feature("glsl", "multiple-render-targets"), "crossgl"
    ).source
    assert "vec4 extra @location(1);" in crossgl and "output.extra = vec4(" in crossgl
    # GLSL ES 3.00 has no noperspective or sample qualifier, and rust-gpu marks only flat inputs.
    absent = {
        "webgl": {"noperspective-interpolation", "sample-interpolation"},
        "rust": set(interfaces.INTERPOLATIONS) - {"flat-interpolation"},
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), set(interfaces.NAMES)),
    }
    for language, names in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & set(interfaces.NAMES) == set(interfaces.NAMES) - names, language


def test_vertex_probes_use_each_languages_built_ins():
    feature = construct_feature("glsl", "clip-distance")
    assert (feature.category, feature.stage) == ("graphics", "vertex")
    assert feature.group == "BuiltIn.ClipDistance Capability.ClipDistance"
    for language, name, line in (
        ("glsl", "vertex-index", "vec4(float(gl_VertexID), 0.0, 0.0, 0.0)"),
        ("webgl", "instance-index", "vec4(float(gl_InstanceID), 0.0, 0.0, 0.0)"),
        ("glsl", "point-size", "gl_PointSize = 2.0;"),
        ("glsl", "clip-distance", "gl_ClipDistance[0] = position.x;"),
        ("webgl", "invariant-position", "invariant gl_Position;\n"),
        ("hlsl", "vertex-index", "uint vid : SV_VertexID"),
        ("hlsl", "clip-distance", "float clip : SV_ClipDistance0;"),
        ("slang", "point-size", "float size : SV_PointSize;"),
        ("metal", "instance-index", "uint iid [[instance_id]]"),
        ("metal", "clip-distance", "float clip [[clip_distance]] [1];"),
        ("metal", "invariant-position", "float4 position [[position, invariant]];"),
        ("wgsl", "vertex-index", "@builtin(vertex_index) vid: u32"),
        ("wgsl", "clip-distance", "@builtin(clip_distances) clip: array<f32, 1>"),
        ("wgsl", "invariant-position", "@builtin(position) @invariant vec4<f32>"),
        ("rust", "point-size", "#[spirv(point_size)] size: &mut f32"),
        ("rust", "invariant-position", "#[spirv(position, invariant)]"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(construct_feature("glsl", "vertex-index"), "crossgl").source
    assert "        int vid @ gl_VertexID;\n" in crossgl and "    vertex {\n" in crossgl
    crossgl = probes.build(
        construct_feature("glsl", "invariant-position"), "crossgl"
    ).source
    assert "vec4 position @ gl_Position @ invariant;" in crossgl
    # Direct3D and WGSL have no point size, HLSL no invariance qualifier and GLSL ES 3.00 no clip distances.
    absent = {
        "hlsl": {"point-size", "invariant-position"},
        "slang": {"invariant-position"},
        "wgsl": {"point-size"},
        "webgl": {"clip-distance"},
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), set(vertices.NAMES)),
    }
    for language, names in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & set(vertices.NAMES) == set(vertices.NAMES) - names, language


def test_draw_parameter_probes_use_each_languages_built_ins():
    feature = construct_feature("glsl", "base-vertex")
    assert (feature.category, feature.stage) == ("graphics", "vertex")
    assert feature.group == "BuiltIn.BaseVertex Capability.DrawParameters"
    for language, name, line in (
        ("glsl", "draw-index", "vec4(float(gl_DrawID), 0.0, 0.0, 0.0)"),
        ("glsl", "base-instance", "vec4(float(gl_BaseInstance), 0.0, 0.0, 0.0)"),
        ("glsl", "cull-distance", "gl_CullDistance[0] = position.x;"),
        ("hlsl", "base-vertex", "int base : SV_StartVertexLocation"),
        ("hlsl", "cull-distance", "float cull : SV_CullDistance0;"),
        ("slang", "draw-index", "uint base : SV_DrawIndex"),
        ("slang", "cull-distance", "float cull[1] : SV_CullDistance;"),
        ("metal", "base-instance", "uint base [[base_instance]]"),
        ("rust", "draw-index", "#[spirv(draw_index)] base: u32"),
        ("rust", "cull-distance", "#[spirv(cull_distance)] cull: &mut [f32; 1]"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(construct_feature("glsl", "base-vertex"), "crossgl").source
    assert (
        "        int base @ gl_BaseVertex;\n" in crossgl and "    vertex {\n" in crossgl
    )
    crossgl = probes.build(construct_feature("glsl", "cull-distance"), "crossgl").source
    assert "float cull @ gl_CullDistance;" in crossgl
    # HLSL has no draw index, Metal neither a draw index nor cull distances, and GLSL ES 3.00 and WGSL none of the four.
    names = set(draws.NAMES)
    absent = {
        "hlsl": {"draw-index"},
        "slang": set(),
        "metal": {"draw-index", "cull-distance"},
        "wgsl": names,
        "webgl": names,
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_rasterization_probes_read_the_primitive_and_sample_or_write_coverage():
    feature = construct_feature("glsl", "sample-index")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    assert feature.group == "BuiltIn.SampleId Capability.SampleRateShading"
    for language, name, line in (
        ("glsl", "primitive-id", "float(gl_PrimitiveID)"),
        ("glsl", "sample-mask-input", "gl_SampleMaskIn[0] & 1"),
        ("glsl", "sample-mask", "gl_SampleMask[0] = 1;"),
        ("hlsl", "sample-index", "uint sid : SV_SampleIndex"),
        ("hlsl", "sample-mask", "uint mask : SV_Coverage;"),
        ("slang", "primitive-id", "uint pid : SV_PrimitiveID"),
        ("metal", "sample-mask-input", "uint mask [[sample_mask]]"),
        ("metal", "sample-mask", "uint mask [[sample_mask]];"),
        ("wgsl", "primitive-id", "enable primitive_index;"),
        ("wgsl", "sample-index", "@builtin(sample_index) sid: u32"),
        ("rust", "sample-mask-input", "#[spirv(sample_mask, flat)] mask: [u32; 1]"),
        ("rust", "primitive-id", "#[spirv(primitive_id, flat)] pid: u32"),
        ("rust", "sample-mask", "#[spirv(sample_mask)] mask: &mut [u32; 1]"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "sample-mask-input"), "crossgl"
    ).source
    assert "        uint mask @ gl_SampleMaskIn;\n" in crossgl
    crossgl = probes.build(construct_feature("glsl", "sample-mask"), "crossgl").source
    assert "output.mask = 1u;" in crossgl and "uint mask @ gl_SampleMask;" in crossgl
    # GLSL ES 3.00 has none of the four built-ins, and the kernel languages have no fragment stage.
    absent = dict.fromkeys(
        ("webgl", "cuda", "hip", "opencl", "mojo"), set(rasterization.NAMES)
    )
    for language, names in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert (
            listed & set(rasterization.NAMES) == set(rasterization.NAMES) - names
        ), language


def test_pipeline_probes_use_each_languages_fragment_built_ins_and_modes():
    feature = construct_feature("glsl", "conservative-depth")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    assert feature.group == "BuiltIn.FragDepth ExecutionMode.DepthGreater"
    assert construct_feature("glsl", "early-fragment-tests").group == (
        "ExecutionMode.EarlyFragmentTests"
    )
    for language, name, line in (
        ("glsl", "helper-invocation", "gl_HelperInvocation ? 1.0 : 0.0"),
        (
            "glsl",
            "early-fragment-tests",
            "layout(early_fragment_tests) in;\nvoid main()",
        ),
        ("glsl", "conservative-depth", "layout(depth_greater) out float gl_FragDepth;"),
        ("webgl", "point-coordinate", "vec4(gl_PointCoord, uv.x, 1.0)"),
        ("hlsl", "helper-invocation", "IsHelperLane() ? 1.0 : 0.0"),
        ("hlsl", "early-fragment-tests", "[earlydepthstencil]\nfloat4 main("),
        ("hlsl", "layer", "uint layer : SV_RenderTargetArrayIndex"),
        ("hlsl", "conservative-depth", "float depth : SV_DepthGreaterEqual;"),
        ("slang", "point-coordinate", "float2 point : SV_PointCoord"),
        (
            "slang",
            "early-fragment-tests",
            '[shader("fragment")]\n[earlydepthstencil]\n',
        ),
        ("metal", "early-fragment-tests", "[[early_fragment_tests]] fragment float4"),
        ("metal", "viewport-index", "uint viewport [[viewport_array_index]]"),
        ("metal", "conservative-depth", "float depth [[depth(greater)]];"),
        ("rust", "helper-invocation", "#[spirv(helper_invocation)] helper: bool"),
        ("rust", "layer", "#[spirv(layer, flat)] layer: u32"),
        ("rust", "early-fragment-tests", "#[spirv(fragment(early_fragment_tests))]"),
        (
            "rust",
            "conservative-depth",
            "#[spirv(fragment(depth_replacing, depth_greater))]",
        ),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "early-fragment-tests"), "crossgl"
    ).source
    assert "    fragment {\n        layout(early_fragment_tests) in;\n" in crossgl
    crossgl = probes.build(
        construct_feature("glsl", "viewport-index"), "crossgl"
    ).source
    assert "        int viewport @ gl_ViewportIndex;\n" in crossgl
    crossgl = probes.build(
        construct_feature("glsl", "conservative-depth"), "crossgl"
    ).source
    assert "    out float gl_FragDepth @depth_greater;\n" in crossgl
    # Direct3D has no point sprites, WGSL none of the six, GLSL ES 3.00 only the point coordinate, and the kernel
    # languages no fragment stage.
    names = set(pipeline.NAMES)
    absent = {
        "hlsl": {"point-coordinate"},
        "wgsl": names,
        "webgl": names - {"point-coordinate"},
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_extension_probes_read_each_languages_fragment_built_ins():
    feature = construct_feature("glsl", "barycentric-coordinate")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    assert feature.group == "BuiltIn.BaryCoordKHR Capability.FragmentBarycentricKHR"
    assert construct_feature("hlsl", "stencil-export").group == (
        "BuiltIn.FragStencilRefEXT Capability.StencilExportEXT"
    )
    for language, name, line in (
        (
            "glsl",
            "barycentric-coordinate",
            "#extension GL_EXT_fragment_shader_barycentric : require\n",
        ),
        ("glsl", "shading-rate", "float(gl_ShadingRateEXT)"),
        ("glsl", "view-index", "#extension GL_EXT_multiview : require\n"),
        ("glsl", "stencil-export", "gl_FragStencilRefARB = int(uv.x * 255.0);"),
        ("webgl", "view-index", "float(gl_ViewID_OVR)"),
        ("hlsl", "barycentric-coordinate", "float3 bary : SV_Barycentrics"),
        ("hlsl", "shading-rate", "uint rate : SV_ShadingRate"),
        ("hlsl", "stencil-export", "uint stencil : SV_StencilRef;"),
        ("slang", "view-index", '[shader("fragment")]\nfloat4 main('),
        (
            "slang",
            "stencil-export",
            '};\n\n[shader("fragment")]\nFragmentOutput main(',
        ),
        ("metal", "barycentric-coordinate", "float3 bary [[barycentric_coord]]"),
        ("metal", "view-index", "uint view [[amplification_id]]"),
        ("metal", "stencil-export", "uint stencil [[stencil]];"),
        ("rust", "view-index", "#[spirv(view_index, flat)] view: u32"),
        ("rust", "stencil-export", "#[spirv(fragment(stencil_ref_replacing_ext))]"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "barycentric-coordinate"), "crossgl"
    ).source
    assert "        vec3 bary @ gl_BaryCoordEXT;\n" in crossgl
    crossgl = probes.build(
        construct_feature("glsl", "stencil-export"), "crossgl"
    ).source
    assert "        int stencil @ gl_FragStencilRefARB;\n" in crossgl
    # Metal has no shading-rate input and rust-gpu no shading-rate built-in, WGSL none of the four, GLSL ES 3.00 only
    # the view index, and the kernel languages no fragment stage.
    names = set(extensions.NAMES)
    absent = {
        "glsl": set(),
        "hlsl": set(),
        "slang": set(),
        "metal": {"shading-rate"},
        "rust": {"shading-rate"},
        "wgsl": names,
        "webgl": names - {"view-index"},
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language
