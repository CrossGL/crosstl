import json
import re
from pathlib import Path

import pytest

from tools.coverage import (
    constructs,
    crossgl,
    execution,
    probes,
    semantic,
    templates,
    validators,
)
from tools.coverage.model import Feature, Result
from tools.coverage.templates import (
    bindings,
    draws,
    extensions,
    fragments,
    interfaces,
    meshes,
    pipeline,
    primitives,
    qualifiers,
    rasterization,
    rays,
    relations,
    specializations,
    subgroups,
    tasks,
    tessellation,
    textures,
    vertices,
)
from tools.coverage.templates.base import ABSENT, OPERATORS

_MODELS = [
    ("Vertex", ["Shader"]),
    ("TessellationControl", ["Tessellation"]),
    ("TessellationEvaluation", ["Tessellation"]),
    ("Geometry", ["Geometry"]),
    ("Fragment", ["Shader"]),
    ("GLCompute", ["Shader"]),
    ("Kernel", ["Kernel"]),
    ("TaskNV", ["MeshShadingNV"]),
    ("TaskEXT", ["MeshShadingEXT"]),
    ("MeshEXT", ["MeshShadingEXT"]),
    ("RayGenerationKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("IntersectionKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("AnyHitKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("ClosestHitKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("MissKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("CallableKHR", ["RayTracingNV", "RayTracingKHR"]),
]


def _grammar(drop: str = "") -> str:
    """A grammar with exactly the items the constructs anchor to, less ``drop``."""
    anchors = {a for c in constructs._curated() for a in c.anchors}
    anchors |= {f"Capability.{name}" for name in constructs._WIDTHS}
    kinds: dict[str, list[dict]] = {
        "Capability": [],
        "StorageClass": [],
        "BuiltIn": [],
        "Decoration": [],
        "ExecutionModel": [],
        "ExecutionMode": [],
        "Dim": [],
    }
    instructions = []
    for anchor in sorted(anchors - {drop}):
        kind, _, name = anchor.rpartition(".")
        if kind:
            kinds[kind].append({"enumerant": name})
        else:
            instructions.append({"opname": name})
    kinds["ExecutionModel"] = [
        {"enumerant": name, "capabilities": caps} for name, caps in _MODELS
    ]
    return json.dumps(
        {
            "instructions": instructions,
            "operand_kinds": [
                {"kind": kind, "enumerants": values} for kind, values in kinds.items()
            ],
        }
    )


GRAMMAR = _grammar()


def _feature(language: str, name: str) -> Feature:
    (feature,) = (f for f in constructs.features(language, GRAMMAR) if f.name == name)
    return feature


def test_tessellation_probes_state_each_languages_domain_spacing_and_patch_values():
    feature = _feature("glsl", "tessellation-domain-quads")
    assert (feature.category, feature.stage) == ("graphics", "tessellation-evaluation")
    assert feature.group == "ExecutionMode.Quads"
    assert _feature("hlsl", "tessellation-control-winding-ccw").stage == (
        "tessellation-control"
    )
    assert _feature("glsl", "tessellation-level-input").group == (
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(_feature("glsl", "tessellation-winding-cw"), "glsl").suffix
        == ".tese"
    )
    crossgl = probes.build(
        _feature("glsl", "tessellation-winding-cw"), "crossgl"
    ).source
    assert "        layout(triangles, equal_spacing, cw) in;\n" in crossgl
    crossgl = probes.build(
        _feature("hlsl", "tessellation-domain-quads"), "crossgl"
    ).source
    assert "OutputPatch<ControlPoint, 4> patch) @domain(quad) {\n" in crossgl
    crossgl = probes.build(
        _feature("hlsl", "tessellation-control-point-mode"), "crossgl"
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
    feature = _feature("glsl", "mesh-output-lines")
    assert (feature.category, feature.stage) == ("graphics", "mesh")
    assert feature.group == (
        "ExecutionMode.OutputLinesEXT BuiltIn.PrimitiveLineIndicesEXT"
    )
    assert _feature("hlsl", "mesh-task-payload").group == (
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert probes.build(_feature("glsl", "mesh-layer"), "glsl").suffix == ".mesh"
    crossgl = probes.build(_feature("glsl", "mesh-output-points"), "crossgl").source
    assert "@ indices out uint points[1])\n" in crossgl
    assert "            @outputtopology(point) {\n" in crossgl
    crossgl = probes.build(_feature("hlsl", "mesh-task-payload"), "crossgl").source
    assert ", @ mesh_payload in Payload payload)\n" in crossgl
    crossgl = probes.build(_feature("hlsl", "mesh-primitive-id"), "crossgl").source
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
    assert _feature("glsl", "mesh-workgroup-id").group == "BuiltIn.WorkgroupId"
    feature = _feature("hlsl", "task-local-invocation-index")
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert probes.build(_feature("glsl", "task-workgroup-id"), "glsl").suffix == ".task"
    crossgl = probes.build(_feature("hlsl", "task-local-invocation-index"), "crossgl")
    assert (
        "void main(uint index @ gl_LocalInvocationIndex) @numthreads(2, 1, 1) {\n"
        in crossgl.source
    )
    crossgl = probes.build(_feature("hlsl", "mesh-workgroup-id"), "crossgl").source
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
    feature = _feature("glsl", "ray-payload")
    assert (feature.category, feature.stage) == ("graphics", "ray-generation")
    assert feature.group == "StorageClass.RayPayloadKHR OpTraceRayKHR"
    assert _feature("hlsl", "ray-custom-attributes").stage == "closest-hit"
    assert _feature("slang", "ray-reported-attributes").stage == "intersection"
    assert _feature("rust", "ray-shader-record").group == (
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(_feature("glsl", "ray-shader-record"), "glsl").suffix == ".rmiss"
    )
    assert (
        probes.build(_feature("glsl", "ray-reported-attributes"), "glsl").suffix
        == ".rint"
    )
    crossgl = probes.build(_feature("hlsl", "ray-custom-attributes"), "crossgl").source
    assert (
        "void main(Payload payload @ payload, Attributes attributes @ hit_attribute) {\n"
        in crossgl
    )
    crossgl = probes.build(_feature("hlsl", "ray-shader-record"), "crossgl").source
    assert (
        "    Record record @glsl_buffer_block(shaderRecordEXT, std430);\n    ray_miss {\n"
        in crossgl
    )
    crossgl = probes.build(_feature("glsl", "ray-payload"), "crossgl").source
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


def test_binding_probes_push_constants_and_bind_a_block_in_another_set():
    feature = _feature("glsl", "push-constant")
    assert (feature.category, feature.target) == ("resource", "vulkan")
    assert feature.group == "StorageClass.PushConstant"
    assert _feature("hlsl", "push-constant-offset").group == (
        "StorageClass.PushConstant Decoration.Offset"
    )
    assert _feature("wgsl", "descriptor-set").group == "Decoration.DescriptorSet"
    assert _feature("glsl", "descriptor-set").target == "vulkan"
    for language, name, line in (
        ("glsl", "push-constant", "layout(push_constant) uniform Constants {\n"),
        ("glsl", "push-constant-offset", "    layout(offset = 16) float scale;\n"),
        (
            "glsl",
            "descriptor-set",
            "layout(std140, set = 1, binding = 3) uniform Params {\n",
        ),
        (
            "hlsl",
            "push-constant",
            "[[vk::push_constant]] ConstantBuffer<Constants> pc;\n",
        ),
        ("hlsl", "push-constant-offset", "    [[vk::offset(16)]] float scale;\n"),
        ("slang", "descriptor-set", "cbuffer Params : register(b3, space1) {\n"),
        (
            "wgsl",
            "descriptor-set",
            "@group(1) @binding(3) var<uniform> params: Params;\n",
        ),
        ("rust", "push-constant", "#[spirv(push_constant)] pc: &Constants"),
        (
            "rust",
            "descriptor-set",
            "#[spirv(uniform, descriptor_set = 1, binding = 3)] scale: &f32",
        ),
        ("glsl", "push-constant", "    outp[i] = fin[i] * pc.scale + pc.bias;\n"),
    ):
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    for name, line in (
        ("push-constant", "    cbuffer Constants @push_constant {\n"),
        ("push-constant-offset", "        float scale @offset(16);\n"),
        ("descriptor-set", "    cbuffer Params @set(1) @binding(3) {\n"),
    ):
        assert line in probes.build(_feature("glsl", name), "crossgl").source, name
    # Metal binds every argument at an index of one table per resource kind, WGSL has no push-constant address
    # space, rust-gpu places no push-constant member, and GLSL ES 3.00 and the kernel languages have neither.
    names = set(bindings.NAMES)
    for language in ("glsl", "hlsl", "slang"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert names <= listed, language
    assert {f.name for f in constructs.features("wgsl", GRAMMAR)} & names == {
        "descriptor-set"
    }
    assert {f.name for f in constructs.features("rust", GRAMMAR)} & names == {
        "push-constant",
        "descriptor-set",
    }
    for language in ("metal", "webgl", "cuda", "hip", "opencl", "mojo"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not listed & names, language


def test_stages_are_the_shader_execution_models():
    stages = [c.name for c in constructs.universe(GRAMMAR) if c.category == "stage"]
    assert stages == [
        "vertex",
        "tessellation-control",
        "tessellation-evaluation",
        "geometry",
        "fragment",
        "compute",
        "task",
        "mesh",
        "ray-generation",
        "intersection",
        "any-hit",
        "closest-hit",
        "miss",
        "callable",
    ]


def test_scalar_widths_come_from_capabilities():
    types = {c.name: c.anchors for c in constructs.universe(GRAMMAR)}
    assert types["int64"] == ("Capability.Int64",)
    assert types["half"] == ("Capability.Float16",)


def test_an_anchor_missing_upstream_fails_loudly():
    with pytest.raises(ValueError, match="OpSwitch"):
        constructs.universe(_grammar(drop="OpSwitch"))


def test_a_language_lists_only_what_it_can_spell():
    wgsl = {f.name for f in constructs.features("wgsl", GRAMMAR)}
    assert {"switch", "half", "vertex"} <= wgsl
    assert not {"double", "int64", "geometry", "any-hit"} & wgsl
    assert "num-workgroups" not in {
        f.name for f in constructs.features("hlsl", GRAMMAR)
    }


def test_spirv_lists_every_construct():
    spirv = constructs.features("spirv", GRAMMAR)
    assert len(spirv) == len(constructs.universe(GRAMMAR))


def test_constructs_share_a_semantic_id_across_languages():
    ids = {
        semantic.lower(_feature(lang, "switch"))[0]
        for lang in ("glsl", "wgsl", "spirv")
    }
    assert ids == {"construct:control-flow/switch"}


def test_every_curated_construct_has_a_crossgl_template():
    assert {c.name for c in constructs._curated()} <= set(crossgl.CONSTRUCTS)


def test_a_construct_without_a_template_is_untested_not_absent(monkeypatch):
    feature = _feature("glsl", "tessellation-control")
    monkeypatch.delitem(templates.TEMPLATES["glsl"], "tessellation-control")
    with pytest.raises(probes.ProbeError, match="no glsl probe template"):
        probes.build(feature, "glsl")


def test_kernel_templates_fill_the_language_spelling_of_types():
    hlsl = probes.build(_feature("hlsl", "swizzle"), "hlsl").source
    assert "float4 v = float4(" in hlsl
    crossgl_probe = probes.build(_feature("glsl", "struct"), "crossgl").source
    assert crossgl_probe.index("struct Pair") < crossgl_probe.index("compute {")


def test_stage_programs_take_the_stage_suffix():
    probe = probes.build(_feature("glsl", "fragment"), "glsl")
    assert probe.suffix == ".frag"
    assert probes.build(_feature("glsl", "geometry"), "glsl").suffix == ".geom"
    assert probes.build(_feature("glsl", "discard"), "glsl").suffix == ".frag"
    assert probes.build(_feature("glsl", "closest-hit"), "glsl").suffix == ".rchit"
    tese = probes.build(_feature("glsl", "tessellation-evaluation"), "glsl")
    assert tese.suffix == ".tese"
    assert probes.build(_feature("glsl", "task"), "glsl").suffix == ".task"
    assert probes.build(_feature("glsl", "mesh"), "glsl").suffix == ".mesh"


def test_slang_names_the_hull_stage_on_its_entry_point():
    hull = probes.build(_feature("slang", "tessellation-control"), "slang").source
    assert '[shader("hull")]\n[domain("tri")]' in hull
    task = probes.build(_feature("slang", "task"), "slang").source
    assert '[shader("amplification")]\n[numthreads(1, 1, 1)]' in task


@pytest.mark.parametrize(
    ("stage", "air", "accepted"),
    [
        ("vertex", "define void @f() {\n}\n", False),
        ("vertex", "!air.vertex = !{!9}\n", True),
        ("compute", "!air.vertex = !{!9}\n", False),
        ("tessellation-evaluation", "!air.vertex = !{!9}\n", False),
        (
            "tessellation-evaluation",
            '!air.vertex = !{!9}\n!17 = !{!"air.patch", !"triangle"}\n',
            True,
        ),
        ("mesh", "!air.mesh = !{!9}\n", True),
        ("task", "!air.mesh = !{!9}\n", False),
        ("task", "!air.object = !{!9}\n", True),
        ("intersection", "!air.kernel = !{!9}\n", False),
        ("intersection", "!air.intersection = !{!9}\n", True),
    ],
)
def test_a_metal_probe_needs_an_entry_point_of_its_stage(
    monkeypatch, stage, air, accepted
):
    def run(arguments, cwd):
        Path(cwd, arguments[arguments.index("-o") + 1]).write_text(air)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "_run", run)
    verdict = validators.validate("metal", "", stage)
    assert verdict.accepted is accepted
    assert accepted or f"no {stage} entry point" in verdict.log


def test_a_metal_intersection_probe_is_a_bounding_box_intersection_function():
    probe = probes.build(_feature("metal", "intersection"), "metal")
    assert "using namespace raytracing;" in probe.source
    assert "[[intersection(bounding_box)]]\nBoxResult probe(" in probe.source


def test_ray_tracing_and_mesh_stages_are_checked_for_vulkan(monkeypatch):
    commands = []

    def run(arguments, cwd):
        commands.append(arguments)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "_run", run)
    validators.validate("glsl", "", "miss")
    validators.validate("glsl", "", "mesh")
    validators.validate("glsl", "", "fragment")
    validators.validate("glsl", "", "fragment", target="vulkan")
    assert commands == [
        ["glslangValidator", "--target-env", "vulkan1.2", "probe.rmiss"],
        ["glslangValidator", "--target-env", "vulkan1.2", "probe.mesh"],
        ["glslangValidator", "probe.frag"],
        ["glslangValidator", "--target-env", "vulkan1.1", "probe.frag"],
    ]


def test_entry_parameters_join_the_kernel_signature():
    wgsl = probes.build(_feature("wgsl", "local-invocation-id"), "wgsl").source
    assert "id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>)" in wgsl
    metal = probes.build(_feature("metal", "sampled-texture"), "metal").source
    assert "texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]]" in metal


@pytest.mark.parametrize(
    "generated, accepted",
    [
        ("void main() { x = ProcessTriTessFactorsAvg(); }", True),
        ("void main() { /* debugBreak() not currently supported for HLSL */; }", False),
    ],
)
def test_a_slang_probe_for_hlsl_is_written_as_hlsl_and_compiled_by_dxc(
    monkeypatch, generated, accepted
):
    commands = []

    def run(arguments, cwd, environment=None, stdout=True):
        commands.append(arguments)
        if arguments[0] == "slangc":
            Path(cwd, arguments[arguments.index("-o") + 1]).write_text(generated)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "_run", run)
    verdict = validators.validate(
        "slang", '[shader("hull")]', "tessellation-control", "main", "hlsl"
    )
    assert verdict.accepted is accepted
    slangc = ["slangc", "probe.slang", "-entry", "main", "-target", "hlsl"]
    assert commands[0] == [*slangc, "-o", "probe.hlsl"]
    if accepted:
        profile = validators.HLSL_PROFILES["tessellation-control"]
        assert commands[1][:5] == ["dxc", "-T", profile, "-E", "main"]
        assert commands[1][-3:] == ["probe.hlsl", "-Fo", "probe.bin"]
    else:
        assert len(commands) == 1
        assert "debugBreak() not currently supported" in verdict.log


@pytest.mark.skipif(not validators.available("slang"), reason="slangc is not installed")
def test_slangc_validates_the_spirv_it_writes():
    # The half-precision atomic converts its pointer to an integer, which SPIR-V allows only for a physical storage
    # buffer pointer; slangc writes the conversion for a local's address and exits 0 unless told to validate.
    source = "\n".join(
        [
            "RWStructuredBuffer<half> outp;",
            '[shader("compute")]',
            "[numthreads(1, 1, 1)]",
            "void main(uint3 tid : SV_DispatchThreadID) {",
            "    half a0 = outp[tid.x];",
            "    half a1;",
            "    InterlockedAddF16Emulated(&a0, a0, a1);",
            "    outp[tid.x] = a1;",
            "}",
        ]
    )
    verdict = validators.validate("slang", source)
    assert not verdict.accepted
    assert verdict.log.startswith("error: line ")
    assert "PhysicalStorageBuffer" in verdict.log


def test_a_slang_probe_for_hlsl_needs_dxc(monkeypatch):
    monkeypatch.setattr(
        validators, "command", lambda name: (name,) if name == "slangc" else ()
    )
    assert validators.missing("slang") == ""
    assert validators.missing("slang", "hlsl") == "dxc"
    assert not validators.available("slang", "hlsl")


def test_hlsl_entry_falls_back_to_the_stage_convention():
    assert (
        validators.hlsl_entry("float4 PSMain() : SV_Target {}", "fragment") == "PSMain"
    )
    assert validators.hlsl_entry("float4 main() : SV_Target {}", "fragment") == "main"


def _templated(language: str) -> list[Feature]:
    return [
        f
        for f in constructs.features(language, GRAMMAR)
        if isinstance(templates.TEMPLATES[language].get(f.name), templates.Template)
    ]


@pytest.mark.parametrize(
    "language",
    [
        "cuda",
        "glsl",
        "hip",
        "hlsl",
        "metal",
        "mojo",
        "opencl",
        "rust",
        "slang",
        "webgl",
        "wgsl",
    ],
)
def test_native_templates_are_valid(language):
    if not validators.available(language):
        pytest.skip(f"{validators.TOOLS[language]} is not installed")
    rejected = []
    for feature in _templated(language):
        probe = probes.build(feature, language)
        verdict = validators.validate(
            language, probe.source, feature.stage, probe.entry, feature.target
        )
        if not verdict.accepted:
            rejected.append(f"{feature.name}: {verdict.log[:200]}")
    assert not rejected


def _reads(source: str) -> set[str]:
    return set(re.findall(r"\b(fin|iin)\b", source))


def test_a_kernel_probe_declares_only_the_input_buffers_it_reads():
    # The harness that runs probes binds one input buffer at slot 0 and the output at slot 1.
    floats = probes.build(_feature("glsl", "conditional-expression"), "glsl").source
    assert _reads(floats) == {"fin"}
    assert "binding = 0) buffer B0 { float fin[]; };" in floats
    assert "binding = 1) buffer B1 { float outp[]; };" in floats
    ints = probes.build(_feature("glsl", "switch"), "glsl").source
    assert _reads(ints) == {"iin"} and "buffer B0 { int iin[]; };" in ints
    both = probes.build(_feature("glsl", "for-loop"), "glsl").source
    assert _reads(both) == {"fin", "iin"} and "buffer B2 { float outp[]; };" in both
    crossgl_probe = probes.build(_feature("glsl", "swizzle"), "crossgl").source
    assert _reads(crossgl_probe) == {"fin"}
    feature = _feature("hlsl", "swizzle")
    native = probes.build(feature, "hlsl").source
    assert execution.executable(
        Result(feature, "output", "supported", sources=(native, native))
    )


def test_a_kernel_template_can_give_the_output_element_type():
    # The output buffer holds float unless the template names another element type, in the language's spelling.
    vectors = probes.build(_feature("glsl", "vector-buffer"), "glsl").source
    assert "buffer B1 { vec4 outp[]; };" in vectors
    pairs = probes.build(_feature("wgsl", "struct-buffer"), "wgsl").source
    assert "array<Pair>" in pairs and "array<f32>" in pairs
    crossgl_probe = probes.build(_feature("glsl", "vector-buffer"), "crossgl").source
    assert "vec4* outp" in crossgl_probe


def test_a_uniform_float_array_reads_one_word_per_row_in_every_language():
    # std140 and a cbuffer give each float its own 16-byte row; languages that pack a float array declare vectors.
    padded = probes.build(_feature("glsl", "uniform-array"), "glsl").source
    assert "float w[4];" in padded and "+ w[k])" in padded
    for language, declaration in (
        ("metal", "float4 w[4];"),
        ("wgsl", "w: array<vec4<f32>, 4>,"),
        ("cuda", "float4 w[4];"),
        ("rust", "w: [Vec4; 4],"),
    ):
        source = probes.build(_feature(language, "uniform-array"), language).source
        assert declaration in source and "w[k].x" in source


def test_a_uniform_struct_is_declared_before_its_block_and_copied_into_a_mutable_local():
    glsl = probes.build(_feature("glsl", "uniform-struct"), "glsl").source
    assert glsl.index("struct Light {") < glsl.index("uniform Params {")
    assert "    Light light;\n    vec4 tint;" in glsl
    for language, copy in (
        ("wgsl", "var l = params.light;"),
        ("rust", "let mut l = params.light;"),
        ("mojo", "var l = params.light\n"),
        ("hlsl", "Light l = light;"),
    ):
        source = probes.build(
            _feature(language, "uniform-struct-copy"), language
        ).source
        assert copy in source
    # Mojo reads a vector's lanes by index.
    mojo = probes.build(_feature("mojo", "uniform-struct-array"), "mojo").source
    assert "params.lights[k].color[0]" in mojo and "var lights: Array[Light, 2]" in mojo


def test_reserved_name_probes_use_their_own_language_s_words_at_every_declaration_and_use():
    crossgl = probes.build(_feature("glsl", "reserved-type-name"), "crossgl").source
    assert crossgl.index("struct type {") < crossgl.index("compute {")
    assert "type make(float x)" in crossgl and "float weigh(type t)" in crossgl
    for language, declaration, use in (
        ("metal", "constant float var = 2.0;", "fin[i] * var + input[k]"),
        ("hlsl", "static const float loop = 2.0;", "fin[i] * loop + type[k]"),
        ("cuda", "__constant__ float var = 2.0f;", "mod[2 - k] * 1024.0f"),
        ("wgsl", "const half: f32 = 2.0;", "fin[i] * half + output[k]"),
        ("mojo", "comptime output: Array[Float32, 3]", "materialize[output]()"),
    ):
        source = probes.build(
            _feature(language, "reserved-global-name"), language
        ).source
        assert declaration in source and use in source
    opencl = probes.build(_feature("opencl", "reserved-type-name"), "opencl").source
    assert "} let;" in opencl and "let make(float x)" in opencl
    cuda = probes.build(_feature("cuda", "reserved-member-name"), "cuda").source
    assert (
        "__device__ float total(Rec r)" in cuda
        and "r.fn = (float)(iin[i] & 7);" in cuda
    )
    rust = probes.build(_feature("rust", "reserved-type-name"), "rust").source
    assert "pub struct var {" in rust and "var { a: x * 2.0, b: 3 }" in rust


def test_uniform_matrix_probes_read_every_column_from_its_own_16_byte_row():
    crossgl = probes.build(_feature("glsl", "uniform-matrix3"), "crossgl").source
    assert "cbuffer Params @binding(3) {\n        mat3 m;\n        vec4 v;" in crossgl
    # Metal, WGSL and Rust give a 2x2 matrix an 8-byte column stride, so they build it from 16-byte vectors.
    for language, member, build in (
        (
            "metal",
            "float4 c[2];",
            "float2x2 m = float2x2(params.c[0].xy, params.c[1].xy);",
        ),
        (
            "wgsl",
            "c: array<vec4<f32>, 2>,",
            "let m = mat2x2<f32>(params.c[0].xy, params.c[1].xy);",
        ),
        (
            "rust",
            "c: [Vec4; 2],",
            "let m = Mat2::from_cols(Vec2::new(params.c[0].x, params.c[0].y)",
        ),
    ):
        source = probes.build(_feature(language, "uniform-matrix2"), language).source
        assert member in source and build in source, language
    rust = probes.build(_feature("rust", "uniform-matrix3"), "rust").source
    assert "c: [Vec4; 3],\n    v: Vec4," in rust and "m.z_axis.y * params.v.w" in rust
    # HLSL names CrossGL's mat2x3 float3x2 and swaps the indices.
    hlsl = probes.build(_feature("hlsl", "uniform-non-square-matrix"), "hlsl").source
    assert "float3x2 m;" in hlsl and "mul(v.xyz, m)" in hlsl and "m[2][1] * v.w" in hlsl
    wgsl = probes.build(_feature("wgsl", "uniform-non-square-matrix"), "wgsl").source
    assert "m: mat2x3<f32>," in wgsl and "params.m[1][2] * params.v.w" in wgsl
    for language in ("cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("uniform-matrix")
            or f.name == "uniform-non-square-matrix"
        ], language


def test_uniform_bool_probes_store_each_boolean_in_a_32_bit_word():
    crossgl = probes.build(_feature("glsl", "uniform-bool-vector"), "crossgl").source
    assert (
        "cbuffer Params @binding(3) {\n        bvec4 f;\n        bvec2 g;\n        bool h;"
        in crossgl
    )
    hlsl = probes.build(_feature("hlsl", "uniform-bool-vector"), "hlsl").source
    assert "bool4 f;\n    bool2 g;" in hlsl and "bool4 c = f;" in hlsl
    # Metal's bool is one byte and the others cannot put a boolean in a host-written block, so they test a uint.
    for language, member, test in (
        ("metal", "uint b0;", "bool b0 = bool(params.b0);"),
        ("wgsl", "b0: u32,", "let b0 = bool(params.b0);"),
        ("cuda", "unsigned int b0;", "bool b0 = params.b0 != 0u;"),
        ("opencl", "uint b0;", "bool b0 = params.b0 != 0u;"),
        ("rust", "b0: u32,", "let b0 = params.b0 != 0;"),
        ("mojo", "var b0: UInt32", "var b0 = params.b0 != 0"),
    ):
        source = probes.build(_feature(language, "uniform-bool"), language).source
        assert member in source and test in source, language
    for language, member, test in (
        ("metal", "uint4 f;\n    uint2 g;", "bool4 f = bool4(params.f);"),
        ("wgsl", "f: vec4<u32>,\n    g: vec2<u32>,", "var c = vec4<bool>(params.f);"),
        ("rust", "f: UVec4,\n    g: UVec2,", "params.f.cmpne(UVec4::ZERO)"),
        ("mojo", "var f: SIMD[DType.uint32, 4]", "params.f.cast[DType.bool]()"),
    ):
        source = probes.build(
            _feature(language, "uniform-bool-vector"), language
        ).source
        assert member in source and test in source, language
    for language in ("cuda", "hip", "opencl"):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert "uniform-bool" in names and "uniform-bool-vector" not in names, language


def test_row_major_probes_store_each_matrix_row_on_its_own_16_byte_row():
    crossgl = probes.build(
        _feature("glsl", "uniform-row-major-non-square"), "crossgl"
    ).source
    assert (
        "cbuffer Params @binding(3) {\n        @ row_major mat2x3 m;\n        vec4 v;"
        in crossgl
    )
    glsl = probes.build(_feature("glsl", "uniform-row-major-array"), "glsl").source
    assert "layout(row_major) mat2 m[2];" in glsl and "m[0][1][0] * 8.0" in glsl
    # HLSL indexes by rows, so CrossGL's element in column 1 and row 2 is m[2][1].
    hlsl = probes.build(_feature("hlsl", "uniform-row-major-non-square"), "hlsl").source
    assert "row_major float3x2 m;\n    float4 v;" in hlsl and "m[2][1] * v.w" in hlsl
    # Metal, WGSL and Rust have no row-major matrices, so they transpose the matrix whose columns are the rows.
    for language, member, build in (
        (
            "metal",
            "float4 r[3];\n    float4 v;",
            "float3x3 m = transpose(float3x3(params.r[0].xyz,",
        ),
        (
            "wgsl",
            "r: array<vec4<f32>, 3>,\n    v: vec4<f32>,",
            "let m = transpose(mat3x3<f32>(params.r[0].xyz,",
        ),
        (
            "rust",
            "r: [Vec4; 3],\n    v: Vec4,",
            ".transpose();\n    let t = m * params.v",
        ),
    ):
        source = probes.build(
            _feature(language, "uniform-row-major-matrix"), language
        ).source
        assert member in source and build in source, language
    names = {f.name for f in constructs.features("rust", GRAMMAR)}
    assert "uniform-row-major-array" in names
    assert "uniform-row-major-non-square" not in names
    for language in ("cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("uniform-row-major")
        ], language


def test_buffer_row_major_probes_store_struct_matrices_by_rows():
    crossgl = probes.build(
        _feature("glsl", "buffer-row-major-non-square"), "crossgl"
    ).source
    assert "struct Rows {\n        @ row_major mat2x3 m;\n        vec2 w;" in crossgl
    assert "buffer Rows* outp;" in crossgl
    # GLSL has no layout qualifier on a structure member, so the buffer's array is row-major.
    glsl = probes.build(_feature("glsl", "buffer-row-major-array"), "glsl").source
    assert "{ layout(row_major) Rows outp[]; }" in glsl and "mat2 m[2];" in glsl
    hlsl = probes.build(_feature("hlsl", "buffer-row-major-matrix"), "hlsl").source
    assert "row_major float2x2 m;" in hlsl and "RWStructuredBuffer<Rows>" in hlsl
    assert "outp[i].m[1][0] += 0.25;" in hlsl
    # Metal, WGSL and Rust store the rows of the matrix's transpose.
    for language, member, store in (
        ("metal", "float2 r[3];", "outp[i].r[2][1] += 0.25;"),
        ("wgsl", "r: array<vec2<f32>, 3>,", "let t = transpose(mat2x3<f32>("),
    ):
        source = probes.build(
            _feature(language, "buffer-row-major-non-square"), language
        ).source
        assert member in source and store in source, language
    rust = probes.build(_feature("rust", "buffer-row-major-array"), "rust").source
    assert "r: [[Vec2; 2]; 2]," in rust and "outp[i].r[k][0].y += 0.25;" in rust
    names = {f.name for f in constructs.features("rust", GRAMMAR)}
    assert "buffer-row-major-matrix" in names
    assert "buffer-row-major-non-square" not in names
    for language in ("webgl", "cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("buffer-row-major")
        ], language


def test_vector_probes_shift_by_scalars_and_resize_with_constructors():
    crossgl = probes.build(_feature("glsl", "vector-shift"), "crossgl").source
    assert "ivec2 r = a >> 1;" in crossgl and "uvec2 w = u << 3u;" in crossgl
    # WGSL shifts by an unsigned vector of the operand's width and has no scalar operand on a bitwise operator.
    wgsl = probes.build(_feature("wgsl", "vector-shift"), "wgsl").source
    assert "let r = a >> vec2<u32>(1u);" in wgsl
    wgsl = probes.build(_feature("wgsl", "vector-bitwise"), "wgsl").source
    assert "(a ^ vec2<i32>(9))" in wgsl and "~a & vec2<i32>(15)" in wgsl
    crossgl = probes.build(_feature("glsl", "matrix-resize"), "crossgl").source
    assert "mat2 a = mat2(m);" in crossgl and "mat2x3 n = mat2x3(m);" in crossgl
    # HLSL casts and indexes by rows; Slang casts only matrices; Metal takes swizzles and columns.
    hlsl = probes.build(_feature("hlsl", "matrix-resize"), "hlsl").source
    assert "float2x2 a = (float2x2)m;" in hlsl and "n[2][1] * 8192.0" in hlsl
    for language, line in (
        ("hlsl", "float2 a = (float2)v;"),
        ("slang", "float2 a = v.xy;"),
        ("metal", "int2 j = k.xy;"),
        ("rust", "let b = v.truncate();"),
    ):
        source = probes.build(_feature(language, "vector-truncation"), language).source
        assert line in source, language
    metal = probes.build(_feature("metal", "matrix-resize"), "metal").source
    assert "float2x2 a = float2x2(m[0].xy, m[1].xy);" in metal
    rust = probes.build(_feature("rust", "matrix-resize"), "rust").source
    assert "Mat3::from_mat2(a * 2.0)" in rust and "n[1].z * 8192.0" in rust
    for language, absent in (
        ("cuda", {"vector-shift", "vector-truncation", "matrix-resize"}),
        ("opencl", {"matrix-resize"}),
        ("mojo", {"vector-truncation", "matrix-resize"}),
    ):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not names & absent, language


def test_fragment_probes_use_each_languages_built_ins():
    # The built-ins are graphics features of the fragment stage, which the kernel languages do not have.
    feature = _feature("glsl", "front-facing")
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, language
    assert probes.build(_feature("glsl", "fragment-depth"), "glsl").suffix == ".frag"
    hlsl = probes.build(_feature("hlsl", "fragment-depth"), "hlsl").source
    assert hlsl.startswith("struct FragmentOutput {") and "[shader(" not in hlsl
    crossgl = probes.build(_feature("glsl", "fragment-depth"), "crossgl").source
    assert "        float depth @ gl_FragDepth;\n" in crossgl
    assert "output.depth = input.uv.x * 0.5 + 0.25;" in crossgl
    crossgl = probes.build(_feature("glsl", "front-facing"), "crossgl").source
    assert "bool front @ gl_FrontFacing;" in crossgl and "input.front ? " in crossgl
    for language in ("cuda", "hip", "opencl", "mojo"):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not names & set(fragments.NAMES), language


def test_interface_probes_qualify_the_fragment_input_or_add_a_second_target():
    feature = _feature("glsl", "sample-interpolation")
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "flat-interpolation"), "crossgl").source
    assert "        vec2 uv @ flat;\n" in crossgl
    crossgl = probes.build(
        _feature("glsl", "multiple-render-targets"), "crossgl"
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
    feature = _feature("glsl", "clip-distance")
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "vertex-index"), "crossgl").source
    assert "        int vid @ gl_VertexID;\n" in crossgl and "    vertex {\n" in crossgl
    crossgl = probes.build(_feature("glsl", "invariant-position"), "crossgl").source
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
    feature = _feature("glsl", "base-vertex")
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "base-vertex"), "crossgl").source
    assert (
        "        int base @ gl_BaseVertex;\n" in crossgl and "    vertex {\n" in crossgl
    )
    crossgl = probes.build(_feature("glsl", "cull-distance"), "crossgl").source
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
    feature = _feature("glsl", "sample-index")
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "sample-mask-input"), "crossgl").source
    assert "        uint mask @ gl_SampleMaskIn;\n" in crossgl
    crossgl = probes.build(_feature("glsl", "sample-mask"), "crossgl").source
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
    feature = _feature("glsl", "conservative-depth")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    assert feature.group == "BuiltIn.FragDepth ExecutionMode.DepthGreater"
    assert _feature("glsl", "early-fragment-tests").group == (
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "early-fragment-tests"), "crossgl").source
    assert "    fragment {\n        layout(early_fragment_tests) in;\n" in crossgl
    crossgl = probes.build(_feature("glsl", "viewport-index"), "crossgl").source
    assert "        int viewport @ gl_ViewportIndex;\n" in crossgl
    crossgl = probes.build(_feature("glsl", "conservative-depth"), "crossgl").source
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
    feature = _feature("glsl", "barycentric-coordinate")
    assert (feature.category, feature.stage) == ("graphics", "fragment")
    assert feature.group == "BuiltIn.BaryCoordKHR Capability.FragmentBarycentricKHR"
    assert _feature("hlsl", "stencil-export").group == (
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "barycentric-coordinate"), "crossgl").source
    assert "        vec3 bary @ gl_BaryCoordEXT;\n" in crossgl
    crossgl = probes.build(_feature("glsl", "stencil-export"), "crossgl").source
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


def test_geometry_probes_state_each_languages_primitives_instances_and_streams():
    feature = _feature("glsl", "geometry-input-lines")
    assert (feature.category, feature.stage) == ("graphics", "geometry")
    assert feature.group == "ExecutionMode.InputLines"
    assert _feature("hlsl", "geometry-invocations").group == (
        "ExecutionMode.Invocations BuiltIn.InvocationId"
    )
    assert _feature("slang", "geometry-streams").group == (
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
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    assert (
        probes.build(_feature("glsl", "geometry-input-lines"), "glsl").suffix == ".geom"
    )
    crossgl = probes.build(
        _feature("glsl", "geometry-input-triangles"), "crossgl"
    ).source
    assert "    in vec4 inPosition[3] @location(0);\n" in crossgl
    assert "        layout(triangles) in;\n" in crossgl
    crossgl = probes.build(_feature("glsl", "geometry-streams"), "crossgl").source
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


def test_subgroup_probes_read_each_languages_built_ins():
    feature = _feature("glsl", "subgroup-invocation-id")
    assert (feature.category, feature.stage) == ("compute", "compute")
    assert feature.group == "BuiltIn.SubgroupLocalInvocationId"
    for language, name, line in (
        ("glsl", "subgroup-size", "#extension GL_KHR_shader_subgroup_basic : require"),
        ("glsl", "subgroup-id", "outp[i] = float(gl_SubgroupID);"),
        ("glsl", "workgroup-size", "outp[i] = float(gl_WorkGroupSize.x);"),
        ("hlsl", "subgroup-invocation-id", "outp[i] = float(WaveGetLaneIndex());"),
        ("slang", "num-subgroups", "outp[i] = float(WaveGetNumWaves());"),
        ("slang", "workgroup-size", "outp[i] = float(WorkgroupSize().x);"),
        ("metal", "subgroup-size", "uint width [[threads_per_simdgroup]]"),
        ("metal", "subgroup-id", "uint group [[simdgroup_index_in_threadgroup]]"),
        ("metal", "workgroup-size", "uint3 size [[threads_per_threadgroup]]"),
        (
            "wgsl",
            "subgroup-invocation-id",
            "@builtin(subgroup_invocation_id) lane: u32",
        ),
        (
            "rust",
            "subgroup-invocation-id",
            "#[spirv(subgroup_local_invocation_id)] lane: u32",
        ),
        ("cuda", "subgroup-invocation-id", "outp[i] = float(index % warpSize);"),
        ("hip", "subgroup-invocation-id", "outp[i] = float(__lane_id());"),
        ("opencl", "num-subgroups", "outp[i] = (float)get_num_sub_groups();"),
        ("mojo", "subgroup-id", "outp[i] = Float32(warp_id())"),
    ):
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "num-subgroups"), "crossgl").source
    assert "outp[i] = float(gl_NumSubgroups);" in crossgl
    # naga accepts the subgroup built-ins only without the enable directive it has not implemented.
    wgsl = probes.build(_feature("wgsl", "subgroup-size"), "wgsl").source
    assert "enable subgroups" not in wgsl
    # HLSL has no built-in for the waves in a group or the group's size, WGSL and Rust none for the group's size, and
    # WebGL 2 no compute stage.
    names = set(subgroups.NAMES)
    absent = {
        "hlsl": {"num-subgroups", "subgroup-id", "workgroup-size"},
        "wgsl": {"workgroup-size"},
        "rust": {"workgroup-size"},
        "webgl": names,
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_specialization_probes_declare_each_languages_pipeline_constants():
    feature = _feature("glsl", "specialization-constant")
    assert (feature.category, feature.stage) == ("compute", "compute")
    assert feature.group == "OpSpecConstant Decoration.SpecId"
    # Only GLSL for Vulkan has constant_id, so the GLSL probes are checked for Vulkan and the others for their usual
    # target.
    assert feature.target == "vulkan"
    assert _feature("slang", "specialization-constant").target == ""
    for language, name, line in (
        (
            "glsl",
            "specialization-constant",
            "layout(constant_id = 0) const int scale = 3;",
        ),
        (
            "glsl",
            "specialization-constant-expression",
            "const int doubled = scale * 2 + 1;",
        ),
        (
            "glsl",
            "specialization-workgroup-size",
            "layout(local_size_x_id = 0, local_size_y = 1",
        ),
        (
            "slang",
            "specialization-constant-bool",
            "[vk::constant_id(0)] const bool negate = true;",
        ),
        ("slang", "specialization-workgroup-size", "[numthreads(size, 1, 1)]"),
        (
            "metal",
            "specialization-constant",
            "constant int scale = is_function_constant_defined(scale_value) ? scale_value : 3;",
        ),
        (
            "metal",
            "specialization-constant-bool",
            "constant bool negate_value [[function_constant(0)]];",
        ),
        (
            "wgsl",
            "specialization-constant-expression",
            "override doubled: i32 = scale * 2 + 1;",
        ),
        (
            "wgsl",
            "specialization-workgroup-size",
            "@compute @workgroup_size(size, 1, 1)",
        ),
        (
            "rust",
            "specialization-constant",
            "#[spirv(spec_constant(id = 0, default = 3))] scale: u32",
        ),
    ):
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        _feature("glsl", "specialization-constant-bool"), "crossgl"
    ).source
    assert "const bool negate @constant_id(0) = true;" in crossgl
    # HLSL for Direct3D, GLSL ES 3.00 and the kernel languages have no specialization constants, Metal no specialized
    # threadgroup size, and rust-gpu only u32 parameters.
    names = set(specializations.NAMES)
    absent = {
        "glsl": set(),
        "slang": set(),
        "wgsl": set(),
        "metal": {"specialization-workgroup-size"},
        "rust": names - {"specialization-constant"},
        **dict.fromkeys(("hlsl", "webgl", "cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_qualifier_probes_give_their_buffers_each_languages_memory_qualifiers():
    feature = _feature("glsl", "coherent-buffer")
    assert (feature.category, feature.stage) == ("compute", "compute")
    assert feature.group == "StorageClass.StorageBuffer Decoration.Coherent"
    for language, name, line in (
        ("glsl", "restrict-buffer", "layout(std430, binding = 0) restrict buffer B0"),
        (
            "glsl",
            "write-only-buffer",
            "layout(std430, binding = 1) writeonly buffer B1",
        ),
        ("hlsl", "coherent-buffer", "globallycoherent RWStructuredBuffer<float> outp"),
        (
            "slang",
            "coherent-buffer",
            "globallycoherent RWStructuredBuffer<float> outp;",
        ),
        ("metal", "restrict-buffer", "device float* __restrict outp [[buffer(1)]]"),
        ("metal", "coherent-buffer", "coherent(device) device float* outp"),
        ("metal", "volatile-buffer", "volatile device float* outp"),
        (
            "cuda",
            "restrict-buffer",
            "float* __restrict__ fin, float* __restrict__ outp",
        ),
        ("hip", "volatile-buffer", "float* fin, volatile float* outp"),
        ("opencl", "restrict-buffer", "__global float* restrict outp"),
        ("opencl", "volatile-buffer", "__global volatile float* outp"),
    ):
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "volatile-buffer"), "crossgl").source
    assert "volatile buffer float* outp;" in crossgl
    # HLSL and Slang have only coherent buffers, Metal no write-only device memory, the kernel languages no coherent
    # or write-only buffers, and WGSL, GLSL ES 3.00, rust-gpu and Mojo no buffer memory qualifiers.
    names = set(qualifiers.NAMES)
    kernels = {"restrict-buffer", "volatile-buffer"}
    absent = {
        "glsl": set(),
        "hlsl": names - {"coherent-buffer"},
        "slang": names - {"coherent-buffer"},
        "metal": {"write-only-buffer"},
        **dict.fromkeys(("cuda", "hip", "opencl"), names - kernels),
        **dict.fromkeys(("wgsl", "webgl", "rust", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_texture_probes_read_each_shape_of_texture_per_language():
    feature = _feature("glsl", "texture-1d")
    assert (feature.category, feature.stage) == ("resource", "compute")
    assert feature.group == "OpTypeImage Dim.1D Capability.Sampled1D"
    assert _feature("hlsl", "texture-cube-array").group == "OpTypeImage Dim.Cube"
    for language, name, line in (
        ("glsl", "texture-1d", "layout(binding = 3) uniform sampler1D tex;"),
        (
            "glsl",
            "texture-cube-array",
            "textureLod(tex, vec4(fin[i], 0.5, 1.0, 1.0), 0.0)",
        ),
        ("glsl", "texel-buffer", "texelFetch(tex, int(fin[i]))"),
        ("webgl", "texture-2d-array", "uniform highp sampler2DArray tex;"),
        ("hlsl", "texture-cube", "TextureCube<float4> tex : register(t1);"),
        ("hlsl", "texel-buffer", "tex.Load(int(fin[i]))"),
        ("slang", "texture-3d", "Texture3D<float4> tex;"),
        ("metal", "texture-1d", "tex.sample(smp, fin[i]).x"),
        ("metal", "texture-2d-array", "texture2d_array<float> tex [[texture(0)]]"),
        ("metal", "texel-buffer", "texture_buffer<float> tex [[texture(0)]]"),
        ("wgsl", "texture-1d", "textureLoad(tex, i32(fin[i]), 0)"),
        ("wgsl", "texture-cube-array", "var tex: texture_cube_array<f32>;"),
        (
            "cuda",
            "texture-2d-array",
            "tex2DLayeredLod<float4>(tex, fin[i], 0.5f, 1, 0.0f)",
        ),
        ("hip", "texel-buffer", "tex1Dfetch<float4>(tex, int(fin[i]))"),
        ("opencl", "texel-buffer", "__read_only image1d_buffer_t tex"),
        ("rust", "texture-cube-array", "&Image!(cube, type=f32, sampled, arrayed)"),
        ("rust", "texel-buffer", "OpImageFetch typeof*{color} %image %coordinate"),
    ):
        probe = probes.build(_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(_feature("glsl", "texture-cube"), "crossgl").source
    assert "uniform samplerCube tex;" in crossgl
    # WGSL has no buffer textures, OpenCL C no cube images, GLSL ES 3.00 no 1D, cube-array or buffer textures, and
    # Mojo's kernels no textures.
    names = set(textures.NAMES)
    absent = {
        **dict.fromkeys(
            ("glsl", "hlsl", "slang", "metal", "cuda", "hip", "rust"), set()
        ),
        "wgsl": {"texel-buffer"},
        "opencl": {"texture-cube", "texture-cube-array"},
        "webgl": {"texture-1d", "texture-cube-array", "texel-buffer"},
        "mojo": names,
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_relation_probes_compare_integer_and_boolean_vectors_per_language():
    # CrossGL and the C family compare with operators; GLSL spells each comparison and negation as a function.
    crossgl = probes.build(
        _feature("glsl", "integer-vector-comparison"), "crossgl"
    ).source
    assert "bvec2 gt = u > uvec2(5u, 7u);" in crossgl and "if (all(ne)) {" in crossgl
    glsl = probes.build(_feature("glsl", "integer-vector-comparison"), "glsl").source
    assert "bvec2 gt = greaterThan(u, uvec2(5u, 7u));" in glsl
    glsl = probes.build(_feature("glsl", "bool-vector-logic"), "glsl").source
    assert "bvec2 n = not(b);" in glsl and "if (any(not(c))) {" in glsl
    crossgl = probes.build(_feature("glsl", "bool-vector-logic"), "crossgl").source
    assert "bvec2 e = b == c;" in crossgl and "if (any(!c)) {" in crossgl
    for language, line in (
        ("hlsl", "bool2 b = (bool2)k;"),
        ("metal", "bool2 b = bool2(k);"),
        ("wgsl", "let u = vec2<u32>(vec2<bool>(fin[i] > 1.0, true));"),
        ("rust", "let g = Vec2::select(b, Vec2::ONE, Vec2::ZERO);"),
        ("mojo", "var j = c.cast[DType.int32]()"),
    ):
        source = probes.build(
            _feature(language, "bool-vector-conversion"), language
        ).source
        assert line in source, language
    opencl = probes.build(
        _feature("opencl", "integer-vector-comparison"), "opencl"
    ).source
    assert "int2 gt = u > (uint2)(5u, 7u);" in opencl and "r += 128.0f;" in opencl
    rust = probes.build(_feature("rust", "bool-vector-logic"), "rust").source
    assert "let e = !(b ^ c);" in rust and "if (!c).any() {" in rust
    for language, absent in (
        ("cuda", set(relations.NAMES)),
        ("opencl", {"bool-vector-logic", "bool-vector-conversion"}),
    ):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not names & absent, language


def test_packing_probes_name_the_layout_rule_and_declare_its_offsets():
    for name, rule in (
        ("uniform-std140-vector", "std140"),
        ("uniform-cbuffer-vector", "hlsl_cbuffer"),
        ("uniform-std140-array", "std140"),
    ):
        crossgl = probes.build(_feature("glsl", name), "crossgl").source
        assert f"cbuffer Params @binding(3) @buffer_layout({rule}) {{" in crossgl, name
    # slangc lays out a plain cbuffer by std140 for SPIR-V, so HLSL and Slang name every member's offset.
    for language in ("hlsl", "slang"):
        std140 = probes.build(_feature(language, "uniform-std140-vector"), language)
        assert (
            "float3 b : packoffset(c1);\n    float c : packoffset(c1.w);"
            in std140.source
        )
        cbuffer = probes.build(_feature(language, "uniform-cbuffer-vector"), language)
        assert (
            "float3 b : packoffset(c0.y);\n    float c : packoffset(c1);"
            in cbuffer.source
        )
        array = probes.build(_feature(language, "uniform-std140-array"), language)
        assert "float d : packoffset(c2);" in array.source
    # No GLSL, WGSL or OpenCL C rule puts a three-component vector at offset 4, so those declare its components.
    for language, member, build in (
        ("glsl", "float b0;", "vec3 b = vec3(b0, b1, b2);"),
        ("wgsl", "b0: f32,", "let b = vec3<f32>(params.b0, params.b1, params.b2);"),
        (
            "opencl",
            "float b0;",
            "float3 b = (float3)(params.b0, params.b1, params.b2);",
        ),
    ):
        source = probes.build(_feature(language, "uniform-cbuffer-vector"), language)
        assert member in source.source and build in source.source, language
    # The languages without std140 rules declare its padding.
    for language, padded in (
        ("metal", "float pad2;\n    packed_float3 b;"),
        ("cuda", "float pad2;\n    float3 b;"),
        ("rust", "pad2: f32,\n    b: Vec3,"),
        ("mojo", "var pad2: Float32\n    var b0: Float32"),
    ):
        source = probes.build(_feature(language, "uniform-std140-vector"), language)
        assert padded in source.source and "pad3" in source.source, language


def test_uniform_integer_probes_keep_each_member_s_signedness_in_every_language():
    crossgl = probes.build(_feature("glsl", "uniform-integer"), "crossgl").source
    assert "cbuffer Params @binding(3) {\n        int n0;" in crossgl
    for language, member, use in (
        ("wgsl", "u1: u32,", "f32(params.u2 % 1000u + (params.u3 >> 24))"),
        ("cuda", "unsigned int u1;", "(float)(params.u2 % 1000u + (params.u3 >> 24))"),
        ("opencl", "uint u1;", "(float)(params.u2 % 1000u + (params.u3 >> 24))"),
        ("rust", "u1: u32,", "(params.u2 % 1000 + (params.u3 >> 24)) as f32"),
        ("mojo", "var u1: UInt32", "Float32(params.u2 % 1000 + (params.u3 >> 24))"),
    ):
        source = probes.build(_feature(language, "uniform-unsigned"), language).source
        assert member in source and use in source, language
    mojo = probes.build(_feature("mojo", "uniform-integer"), "mojo").source
    assert "params.n3 // 1000000" in mojo
    for language, local in (
        ("wgsl", "let v = params.iv;"),
        ("mojo", "var v = params.iv"),
    ):
        vector = probes.build(_feature(language, "uniform-integer-vector"), language)
        assert local in vector.source and "$" not in vector.source, language
    assert "(v[0] >> 20) + (v[1] >> 20) + (v[3] >> 24)" in vector.source


# Each HLSL matrix probe with its matrices written as CrossGL lists them, by columns, and transposed into HLSL's rows.
# The indexing probe keeps CrossGL's columns as HLSL rows instead, and indexes them as CrossGL does.
_HLSL_BY_COLUMNS = {
    "matrix-indexing": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = float2x2(f, 1.0, 2.0, 3.0);",
            "float2 c = m[iin[i] & 1];",
            "outp[i] = m[1][0] * 100.0 + m[0].y * 10.0 + c.x + 0.5 * c.y;",
        ],
    ),
    "matrix-from-vectors": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(float2(f, 1.0), float2(2.0, 3.0)));",
            "outp[i] = m[1][0] * 100.0 + m[0][1] * 10.0 + m[0][0];",
        ],
    ),
    "matrix-vector-product": (
        [("float", "fin"), ("float", "outp")],
        [
            "float2x2 m = transpose(float2x2(fin[i], 1.0, 2.0, 3.0));",
            "float2 v = mul(m, float2(1.0, fin[i]));",
            "outp[i] = v.y;",
        ],
    ),
    "matrix-product": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(f, 1.0, 2.0, 3.0));",
            "float2x2 n = transpose(float2x2(1.0, f, 0.5, 2.0));",
            "float2 v = mul(mul(m, n), float2(1.0, f));",
            "outp[i] = v.x + 2.0 * v.y;",
        ],
    ),
    "vector-matrix-product": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(f, 1.0, 2.0, 3.0));",
            "float2 v = mul(float2(1.0, f), m);",
            "outp[i] = v.x + 2.0 * v.y;",
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(_HLSL_BY_COLUMNS))
def test_the_hlsl_matrix_probes_compute_what_the_crossgl_probes_compute(name):
    # HLSL fills and indexes a matrix by rows and CrossGL by columns, so the probes must write them differently.
    if not validators.can_lower("hlsl"):
        pytest.skip("dxc or spirv-dis is not installed")
    feature = _feature("hlsl", name)
    native = probes.build(feature, "hlsl").source
    by_columns = probes.DIALECTS["hlsl"].kernel(*_HLSL_BY_COLUMNS[name])
    probe = Result(feature, "output", "supported", sources=(native, by_columns))
    [result], used = execution.run([probe], workers=1)
    if not used:
        pytest.skip("no Vulkan runtime")
    assert (result.outcome, result.runtime) == ("supported", used[0])


# glslang folds the swizzle probe's v.wzyx.x to one OpCompositeExtract.
_UNANCHORED_TYPES = {"swizzle"}
# GLSL has no float remainder with the dividend's sign, so its probe computes one through trunc, and no typedef,
# enumeration, member function or boolean bitwise operator; DXC lowers the HLSL probes of all five.
_HLSL_ANCHORED = {
    "float-remainder",
    "type-alias",
    "enum-constant",
    "struct-method",
    "bool-bitwise",
}


def _missing_anchors(language: str, names: set[str] | None = None) -> list[str]:
    """The anchors of the operator and type constructs (or of ``names``) that ``language``'s probe lowers without."""
    missing = []
    for construct in constructs.universe(GRAMMAR):
        if construct.category not in ("operator", "type"):
            continue
        if names is None and construct.name in _UNANCHORED_TYPES | _HLSL_ANCHORED:
            continue
        if names is not None and construct.name not in names:
            continue
        probe = probes.build(_feature(language, construct.name), language)
        assembly, log = validators.lower(
            language, probe.source, probe.entry, "compute", raw_ids=False
        )
        assert assembly, log
        found = set(re.findall(r"\bOp\w+", assembly))
        missing += [
            f"{construct.name}: {a}"
            for a in construct.anchors
            if a.startswith("Op") and a not in found
        ]
    return missing


def test_hlsl_anchored_probes_compile_to_their_anchors():
    if not validators.can_lower("hlsl"):
        pytest.skip("dxc or spirv-dis is not installed")
    assert not _missing_anchors("hlsl", _HLSL_ANCHORED)


def test_operator_and_type_probes_compile_to_their_anchors():
    # glslang lowers the GLSL probe of each operator and type to the instructions the construct is anchored to; it
    # writes a signed remainder as OpSMod, which the probes' non-negative operands make equal to OpSRem. Scalar
    # widths are anchored to capabilities, which this leaves out.
    if not validators.can_lower("glsl"):
        pytest.skip("glslangValidator or spirv-dis is not installed")
    assert not _missing_anchors("glsl")


def test_c_operator_probes_cast_rather_than_construct():
    # CUDA, HIP and OpenCL C have no constructor casts, and CUDA spells uint as unsigned int.
    for language in ("cuda", "hip", "opencl"):
        table = templates.TEMPLATES[language]
        bodies = [table[name].body for name in OPERATORS if table[name] is not ABSENT]
        assert not [
            b for b in bodies if re.search(r"\b(float|int|uint)\(", b)
        ], language
    unsigned = templates.TEMPLATES["cuda"]["unsigned-division"].body
    assert unsigned.startswith("unsigned int a = (unsigned int)iin[i] + 7u;")
