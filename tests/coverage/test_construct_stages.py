"""Probes for the geometry, tessellation, mesh, task and ray-tracing stages."""

from tools.coverage import constructs, probes
from tools.coverage.templates import (
    meshes,
    primitives,
    rays,
    tasks,
    tessellation,
)

from .construct_grammar import GRAMMAR, construct_feature


def test_tessellation_probes_state_each_languages_domain_spacing_and_patch_values():
    feature = construct_feature("glsl", "tessellation-domain-quads")
    assert (feature.category, feature.stage) == ("graphics", "tessellation-evaluation")
    assert feature.group == "ExecutionMode.Quads"
    assert construct_feature("hlsl", "tessellation-control-winding-ccw").stage == (
        "tessellation-control"
    )
    assert construct_feature("glsl", "tessellation-level-input").group == (
        "BuiltIn.TessLevelOuter BuiltIn.TessLevelInner"
    )
    for language, name, line in (
        (
            "glsl",
            "tessellation-domain-isolines",
            "layout(isolines, equal_spacing, ccw) in;\n",
        ),
        (
            "glsl",
            "tessellation-spacing-fractional-odd",
            "layout(triangles, fractional_odd_spacing, ccw) in;\n",
        ),
        ("glsl", "tessellation-point-mode", "equal_spacing, ccw, point_mode) in;\n"),
        (
            "glsl",
            "tessellation-patch-input",
            "layout(location = 1) patch in vec4 extra;\n",
        ),
        ("glsl", "tessellation-patch-vertices", "* float(gl_PatchVerticesIn);\n"),
        ("glsl", "tessellation-level-input", "position[1] * gl_TessLevelInner[0]"),
        (
            "hlsl",
            "tessellation-domain-quads",
            "    float inside[2] : SV_InsideTessFactor;\n",
        ),
        ("hlsl", "tessellation-domain-quads", "float2 coord : SV_DomainLocation"),
        (
            "hlsl",
            "tessellation-domain-isolines",
            "const OutputPatch<ControlPoint, 2> patch",
        ),
        ("hlsl", "tessellation-level-input", "patch[1].position * constants.inside"),
        ("hlsl", "tessellation-control-point-mode", '[outputtopology("point")]\n'),
        (
            "hlsl",
            "tessellation-patch-output",
            "    output.extra = patch[0].position;\n",
        ),
        (
            "slang",
            "tessellation-domain-quads",
            '[shader("domain")]\n[domain("quad")]\n',
        ),
        (
            "slang",
            "tessellation-control-spacing-fractional-even",
            '[shader("hull")]\n[domain("tri")]\n[partitioning("fractional_even")]\n',
        ),
        ("metal", "tessellation-domain-quads", "[[patch(quad, 4)]]\n"),
        ("metal", "tessellation-patch-input", "    float4 extra [[attribute(1)]];\n"),
        (
            "rust",
            "tessellation-domain-quads",
            "#[spirv(tessellation_evaluation(quads, spacing_equal, vertex_order_ccw))]",
        ),
        (
            "rust",
            "tessellation-level-input",
            "#[spirv(tess_level_inner)] inner: [f32; 2]",
        ),
        (
            "rust",
            "tessellation-control-spacing-fractional-odd",
            "tessellation_control(output_vertices = 3, spacing_fraction_odd, vertex_order_cw)",
        ),
        ("rust", "tessellation-patch-vertices", "#[spirv(patch_vertices)] count: u32"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(
            construct_feature("glsl", "tessellation-winding-cw"), "glsl"
        ).suffix
        == ".tese"
    )
    crossgl = probes.build(
        construct_feature("glsl", "tessellation-winding-cw"), "crossgl"
    ).source
    assert "        layout(triangles, equal_spacing, cw) in;\n" in crossgl
    crossgl = probes.build(
        construct_feature("hlsl", "tessellation-domain-quads"), "crossgl"
    ).source
    assert "OutputPatch<ControlPoint, 4> patch) @domain(quad) {\n" in crossgl
    crossgl = probes.build(
        construct_feature("hlsl", "tessellation-control-point-mode"), "crossgl"
    )
    assert "            @outputtopology(point)\n" in crossgl.source
    # HLSL and Slang state spacing, winding and points only on the hull shader, and GLSL only on the evaluation
    # shader. Metal has quads and per-patch inputs alone, rust-gpu no Patch decoration, and GLSL ES 3.00, WGSL and the
    # kernel languages no tessellation.
    names = set(tessellation.NAMES)
    evaluation_layouts = {
        "tessellation-spacing-fractional-odd",
        "tessellation-spacing-fractional-even",
        "tessellation-winding-cw",
        "tessellation-point-mode",
    }
    hull = {name for name in names if name.startswith("tessellation-control-")}
    absent = {
        "glsl": hull,
        "hlsl": evaluation_layouts | {"tessellation-patch-vertices"},
        "slang": evaluation_layouts | {"tessellation-patch-vertices"},
        "metal": names - {"tessellation-domain-quads", "tessellation-patch-input"},
        "rust": {"tessellation-patch-input", "tessellation-patch-output"},
        **dict.fromkeys(("wgsl", "webgl"), names),
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_mesh_probes_state_each_languages_topology_primitive_values_and_payload():
    feature = construct_feature("glsl", "mesh-output-lines")
    assert (feature.category, feature.stage) == ("graphics", "mesh")
    assert feature.group == (
        "ExecutionMode.OutputLinesEXT BuiltIn.PrimitiveLineIndicesEXT"
    )
    assert construct_feature("hlsl", "mesh-task-payload").group == (
        "StorageClass.TaskPayloadWorkgroupEXT"
    )
    for language, name, line in (
        (
            "glsl",
            "mesh-output-points",
            "layout(points, max_vertices = 1, max_primitives = 1) out;\n",
        ),
        ("glsl", "mesh-output-points", "    gl_PrimitivePointIndicesEXT[0] = 0u;\n"),
        (
            "glsl",
            "mesh-per-primitive-output",
            "layout(location = 0) perprimitiveEXT out vec4 color[];\n",
        ),
        (
            "glsl",
            "mesh-cull-primitive",
            "    gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = true;\n",
        ),
        ("glsl", "mesh-task-payload", "taskPayloadSharedEXT Payload payload;\n"),
        ("hlsl", "mesh-output-lines", '[outputtopology("line")]\n'),
        ("hlsl", "mesh-output-lines", "out indices uint2 lines[1]) {\n"),
        ("hlsl", "mesh-layer", "    uint layer : SV_RenderTargetArrayIndex;\n"),
        ("hlsl", "mesh-cull-primitive", ", out primitives Primitive prims[1]) {\n"),
        ("hlsl", "mesh-task-payload", ", in payload Payload payload) {\n"),
        ("slang", "mesh-output-points", '[shader("mesh")]\n[numthreads(1, 1, 1)]\n'),
        ("slang", "mesh-output-points", "out indices uint points[1]) {\n"),
        ("metal", "mesh-output-lines", "mesh<Vertex, void, 2, 1, topology::line>"),
        ("metal", "mesh-cull-primitive", "    bool culled [[primitive_culled]];\n"),
        (
            "metal",
            "mesh-viewport-index",
            "    uint viewport [[viewport_array_index]];\n",
        ),
        (
            "metal",
            "mesh-task-payload",
            "const object_data Payload& payload [[payload]]",
        ),
        ("rust", "mesh-output-points", "output_primitives_ext = 1, output_points))]"),
        (
            "rust",
            "mesh-primitive-id",
            "#[spirv(primitive_id, per_primitive_ext)] id: &mut [i32; 1]",
        ),
        (
            "rust",
            "mesh-task-payload",
            "#[spirv(task_payload_workgroup_ext)] payload: &Payload",
        ),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(construct_feature("glsl", "mesh-layer"), "glsl").suffix == ".mesh"
    )
    crossgl = probes.build(
        construct_feature("glsl", "mesh-output-points"), "crossgl"
    ).source
    assert "@ indices out uint points[1])\n" in crossgl
    assert "            @outputtopology(point) {\n" in crossgl
    crossgl = probes.build(
        construct_feature("hlsl", "mesh-task-payload"), "crossgl"
    ).source
    assert ", @ mesh_payload in Payload payload)\n" in crossgl
    crossgl = probes.build(
        construct_feature("hlsl", "mesh-primitive-id"), "crossgl"
    ).source
    assert "        uint id @ gl_PrimitiveID;\n" in crossgl
    # DXC accepts no point topology, and WGSL, GLSL ES 3.00 and the kernel languages have no mesh stage.
    names = set(meshes.NAMES)
    absent = {
        "glsl": set(),
        "hlsl": {"mesh-output-points"},
        "slang": set(),
        "metal": set(),
        "rust": set(),
        **dict.fromkeys(("wgsl", "webgl"), names),
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_mesh_and_task_probes_read_their_workgroup_and_invocation_ids():
    assert construct_feature("glsl", "mesh-workgroup-id").group == "BuiltIn.WorkgroupId"
    feature = construct_feature("hlsl", "task-local-invocation-index")
    assert (feature.category, feature.stage) == ("graphics", "task")
    assert feature.group == "BuiltIn.LocalInvocationIndex OpTypeArray"
    for language, name, line in (
        (
            "glsl",
            "mesh-workgroup-id",
            "vec4(float(gl_WorkGroupID.x), 0.0, 0.0, 1.0);\n",
        ),
        ("glsl", "mesh-local-invocation-index", "layout(local_size_x = 3) in;\n"),
        (
            "glsl",
            "mesh-local-invocation-index",
            "    if (gl_LocalInvocationIndex == 0u) {\n",
        ),
        ("glsl", "task-payload-array", "    uint values[2];\n"),
        ("glsl", "task-workgroup-id", "    payload.value = gl_WorkGroupID.x;\n"),
        (
            "glsl",
            "task-local-invocation-index",
            "    payload.values[gl_LocalInvocationIndex] = 1u;\n",
        ),
        ("hlsl", "mesh-workgroup-id", ", uint3 group : SV_GroupID) {\n"),
        ("hlsl", "mesh-local-invocation-index", ", uint index : SV_GroupIndex) {\n"),
        ("hlsl", "task-workgroup-id", "void main(uint3 group : SV_GroupID) {\n"),
        ("hlsl", "task-local-invocation-index", "[numthreads(2, 1, 1)]\n"),
        (
            "slang",
            "task-payload-array",
            '[shader("amplification")]\n[numthreads(1, 1, 1)]\n',
        ),
        (
            "metal",
            "mesh-workgroup-id",
            "uint3 group [[threadgroup_position_in_grid]]",
        ),
        (
            "metal",
            "mesh-local-invocation-index",
            "    output.set_vertex(index, Vertex{float4(float(index), 0.0, 0.0, 1.0)});\n",
        ),
        (
            "metal",
            "task-local-invocation-index",
            "uint index [[thread_index_in_threadgroup]]",
        ),
        ("rust", "mesh-workgroup-id", "#[spirv(workgroup_id)] group: UVec3"),
        ("rust", "task-payload-array", "    pub values: [u32; 2],\n"),
        (
            "rust",
            "task-local-invocation-index",
            "#[spirv(task_ext(threads(2)))]\n",
        ),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(construct_feature("glsl", "task-workgroup-id"), "glsl").suffix
        == ".task"
    )
    crossgl = probes.build(
        construct_feature("hlsl", "task-local-invocation-index"), "crossgl"
    )
    assert (
        "void main(uint index @ gl_LocalInvocationIndex) @numthreads(2, 1, 1) {\n"
        in crossgl.source
    )
    crossgl = probes.build(
        construct_feature("hlsl", "mesh-workgroup-id"), "crossgl"
    ).source
    assert ", uvec3 group @ gl_WorkGroupID)\n" in crossgl
    # WGSL, GLSL ES 3.00 and the kernel languages have no mesh or task stage.
    names = {"mesh-workgroup-id", "mesh-local-invocation-index", *tasks.NAMES}
    for language in ("glsl", "hlsl", "slang", "metal", "rust"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert names <= listed, language
    for language in ("wgsl", "webgl", "cuda", "hip", "opencl", "mojo"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not listed & names, language


def test_ray_tracing_probes_hand_payloads_callable_data_attributes_and_records_over():
    feature = construct_feature("glsl", "ray-payload")
    assert (feature.category, feature.stage) == ("graphics", "ray-generation")
    assert feature.group == "StorageClass.RayPayloadKHR OpTraceRayKHR"
    assert construct_feature("hlsl", "ray-custom-attributes").stage == "closest-hit"
    assert construct_feature("slang", "ray-reported-attributes").stage == "intersection"
    assert construct_feature("rust", "ray-shader-record").group == (
        "StorageClass.ShaderRecordBufferKHR"
    )
    for language, name, line in (
        (
            "glsl",
            "ray-payload",
            "layout(location = 0) rayPayloadEXT Payload payload;\n",
        ),
        ("glsl", "ray-callable-data", "    executeCallableEXT(0u, 0);\n"),
        ("glsl", "ray-reported-attributes", "hitAttributeEXT Attributes attributes;\n"),
        (
            "glsl",
            "ray-custom-attributes",
            "    payload.color = vec4(attributes.normal, 1.0);\n",
        ),
        (
            "glsl",
            "ray-shader-record",
            "layout(shaderRecordEXT, std430) buffer Record {\n",
        ),
        (
            "hlsl",
            "ray-payload",
            "    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);\n",
        ),
        ("hlsl", "ray-payload", "write(caller, anyhit, closesthit, miss);\n"),
        ("hlsl", "ray-callable-data", "    CallShader(0, data);\n"),
        ("hlsl", "ray-reported-attributes", "    ReportHit(1.0, 0, attributes);\n"),
        (
            "hlsl",
            "ray-custom-attributes",
            "void main(inout Payload payload, in Attributes attributes) {\n",
        ),
        (
            "hlsl",
            "ray-shader-record",
            "[[vk::shader_record_ext]] ConstantBuffer<Record> record;\n",
        ),
        (
            "slang",
            "ray-shader-record",
            "[[vk::shader_record]] ConstantBuffer<Record> record;\n",
        ),
        ("rust", "ray-payload", "#[spirv(ray_payload)] payload: &mut Payload"),
        (
            "rust",
            "ray-callable-data",
            "spirv_std::arch::execute_callable::<_, 0>(data);\n",
        ),
        (
            "rust",
            "ray-reported-attributes",
            "#[spirv(hit_attribute)] attributes: &mut Attributes",
        ),
        ("rust", "ray-shader-record", "#[spirv(shader_record_buffer)] record: &Record"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(construct_feature("glsl", "ray-shader-record"), "glsl").suffix
        == ".rmiss"
    )
    assert (
        probes.build(
            construct_feature("glsl", "ray-reported-attributes"), "glsl"
        ).suffix
        == ".rint"
    )
    crossgl = probes.build(
        construct_feature("hlsl", "ray-custom-attributes"), "crossgl"
    ).source
    assert (
        "void main(Payload payload @ payload, Attributes attributes @ hit_attribute) {\n"
        in crossgl
    )
    crossgl = probes.build(
        construct_feature("hlsl", "ray-shader-record"), "crossgl"
    ).source
    assert (
        "    Record record @glsl_buffer_block(shaderRecordEXT, std430);\n    ray_miss {\n"
        in crossgl
    )
    crossgl = probes.build(construct_feature("glsl", "ray-payload"), "crossgl").source
    assert "        uniform accelerationStructureEXT scene;\n" in crossgl
    assert (
        "TraceRay(scene, RAY_FLAG_NONE, 255u, 0u, 1u, 0u, ray, payload);\n" in crossgl
    )
    # Metal has no ray-generation, hit, miss or callable stage and no shader binding table; WGSL, GLSL ES 3.00 and
    # the kernel languages have no ray-tracing stages.
    names = set(rays.NAMES)
    for language in ("glsl", "hlsl", "slang", "rust"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert names <= listed, language
    for language in ("metal", "wgsl", "webgl", "cuda", "hip", "opencl", "mojo"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not listed & names, language


def test_geometry_probes_state_each_languages_primitives_instances_and_streams():
    feature = construct_feature("glsl", "geometry-input-lines")
    assert (feature.category, feature.stage) == ("graphics", "geometry")
    assert feature.group == "ExecutionMode.InputLines"
    assert construct_feature("hlsl", "geometry-invocations").group == (
        "ExecutionMode.Invocations BuiltIn.InvocationId"
    )
    assert construct_feature("slang", "geometry-streams").group == (
        "Decoration.Stream Capability.GeometryStreams"
    )
    for language, name, line in (
        ("glsl", "geometry-input-lines", "layout(lines) in;\n"),
        (
            "glsl",
            "geometry-input-triangles-adjacency",
            "layout(triangles_adjacency) in;\n",
        ),
        (
            "glsl",
            "geometry-output-line-strip",
            "layout(line_strip, max_vertices = 2) out;\n",
        ),
        ("glsl", "geometry-invocations", "layout(points, invocations = 2) in;\n"),
        (
            "glsl",
            "geometry-streams",
            "layout(stream = 1, location = 0) out vec4 extra;\n",
        ),
        ("glsl", "geometry-streams", "    EmitStreamVertex(1);\n"),
        (
            "hlsl",
            "geometry-input-lines-adjacency",
            "void main(lineadj float4 position[4] : POSITION",
        ),
        (
            "hlsl",
            "geometry-output-triangle-strip",
            "inout TriangleStream<GeometryOut> stream",
        ),
        ("hlsl", "geometry-invocations", "[instance(2)]\n[maxvertexcount(1)]\n"),
        ("hlsl", "geometry-invocations", "uint id : SV_GSInstanceID"),
        ("hlsl", "geometry-streams", "inout PointStream<ExtraOut> extras"),
        (
            "slang",
            "geometry-input-triangles",
            '[shader("geometry")]\n[maxvertexcount(1)]\n',
        ),
        ("slang", "geometry-invocations", '[shader("geometry")]\n[instance(2)]\n'),
        (
            "rust",
            "geometry-input-triangles",
            "#[spirv(geometry(triangles = 3, output_points = 1",
        ),
        (
            "rust",
            "geometry-output-line-strip",
            "output_line_strip = 2, output_vertices = 2",
        ),
        ("rust", "geometry-invocations", "#[spirv(invocation_id)] id: u32"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(construct_feature("glsl", "geometry-input-lines"), "glsl").suffix
        == ".geom"
    )
    crossgl = probes.build(
        construct_feature("glsl", "geometry-input-triangles"), "crossgl"
    ).source
    assert "    in vec4 inPosition[3] @location(0);\n" in crossgl
    assert "        layout(triangles) in;\n" in crossgl
    crossgl = probes.build(
        construct_feature("glsl", "geometry-streams"), "crossgl"
    ).source
    assert "    out vec4 extra @location(0) @stream(1);\n" in crossgl
    # rust-gpu has no Stream decoration; Metal, WGSL and GLSL ES 3.00 have no geometry stage, and the kernel languages
    # no graphics stages.
    names = set(primitives.NAMES)
    absent = {
        "glsl": set(),
        "hlsl": set(),
        "slang": set(),
        "rust": {"geometry-streams"},
        **dict.fromkeys(("metal", "wgsl", "webgl"), names),
        **dict.fromkeys(("cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language
