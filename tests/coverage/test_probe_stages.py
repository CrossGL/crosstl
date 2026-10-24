"""Probes for features that need a shader stage other than compute."""

from dataclasses import replace

import pytest

from tools.coverage import probes, validators
from tools.coverage.model import VOID, Feature, Param, Signature, Type

from .glsl_features import VEC3, feature

DFDX = feature("dFdx", VEC3, VEC3, stage="fragment")


@pytest.mark.parametrize(
    ("language", "entry"),
    [
        ("crossgl", "FragmentOutput main(FragmentInput input) {"),
        ("glsl", "layout(location = 0) out vec4 color;"),
        ("hlsl", "float4 main() : SV_Target {"),
        ("metal", "fragment float4 probe(device float* fin [[buffer(0)]]"),
        ("rust", "#[spirv(fragment)]"),
        ("slang", '[shader("fragment")]\nfloat4 main() : SV_Target {'),
        ("wgsl", "@fragment\nfn main() -> @location(0) vec4<f32> {"),
    ],
)
def test_fragment_features_get_a_fragment_shader(language, entry):
    probe = probes.build(DFDX, language)
    assert entry in probe.source
    assert "compute" not in probe.source and "numthreads" not in probe.source
    assert probe.suffix == (
        ".frag" if language == "glsl" else probes.DIALECTS[language].suffix
    )


@pytest.mark.parametrize("language", ["glsl", "hlsl", "metal", "slang", "wgsl"])
def test_fragment_probes_are_valid(language):
    if not validators.available(language):
        pytest.skip(f"no {language} validator")
    name = {"hlsl": "ddx", "metal": "dfdx", "slang": "ddx", "wgsl": "dpdx"}.get(
        language, "dFdx"
    )
    spelled = Feature(
        language, "function", name, DFDX.signature, crossgl="dFdx", stage="fragment"
    )
    for baseline in (False, True):
        probe = probes.build(spelled, language, baseline=baseline)
        verdict = validators.validate(language, probe.source, "fragment", probe.entry)
        assert verdict.accepted, verdict.log


WORLD_RAY_ORIGIN = feature("WorldRayOrigin", VEC3, stage="closest-hit")
HIT_PARAMETERS = "void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {"


@pytest.mark.parametrize(
    ("language", "entry"),
    [
        (
            "crossgl",
            "    ray_closest_hit {\n        buffer vec3* outp;\n        void main(Payload payload @ payload, "
            "BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {",
        ),
        (
            "hlsl",
            f'RWStructuredBuffer<float3> outp : register(u0);\n[shader("closesthit")]\n{HIT_PARAMETERS}',
        ),
        (
            "slang",
            f'RWStructuredBuffer<float3> outp;\n[shader("closesthit")]\n{HIT_PARAMETERS}',
        ),
    ],
)
def test_raytracing_features_get_a_shader_of_their_stage(language, entry):
    probe = probes.build(WORLD_RAY_ORIGIN, language)
    assert entry in probe.source
    assert "numthreads" not in probe.source
    # Payload access qualifiers are required from HLSL shader model 6.7; Slang takes a plain struct.
    payload = {"hlsl": "struct [raypayload] Payload", "slang": "struct Payload\n{"}
    if language in payload:
        assert payload[language] in probe.source


def test_a_raytracing_shader_without_a_hit_takes_no_payload():
    reorder = feature("ReorderThread", VOID, Type("uint"), stage="ray-generation")
    source = probes.build(reorder, "hlsl").source
    assert '[shader("raygeneration")]\nvoid main() {' in source
    assert "Payload" not in source
    assert "    ray_generation {" in probes.build(reorder, "crossgl").source
    with pytest.raises(probes.ProbeError, match="no closest-hit-stage probe for glsl"):
        probes.build(WORLD_RAY_ORIGIN, "glsl")


@pytest.mark.parametrize("language", ["hlsl", "slang"])
def test_raytracing_probes_are_valid(language):
    if not validators.available(language):
        pytest.skip(f"no {language} validator")
    for stage, name in (("closest-hit", "WorldRayOrigin"), ("any-hit", "HitKind")):
        result = Type("uint") if name == "HitKind" else VEC3
        spelled = Feature(
            language, "function", name, Signature(result, ()), crossgl=name, stage=stage
        )
        for baseline in (False, True):
            probe = probes.build(spelled, language, baseline=baseline)
            verdict = validators.validate(language, probe.source, stage, probe.entry)
            assert verdict.accepted, verdict.log


REMAINING_RECURSION = feature("GetRemainingRecursionLevels", Type("uint"), stage="node")


def test_node_features_get_a_broadcasting_node_shader_crossgl_cannot_spell():
    source = probes.build(REMAINING_RECURSION, "hlsl").source
    assert '[Shader("node")]\n[NodeLaunch("broadcasting")]' in source
    assert "    uint r = GetRemainingRecursionLevels();" in source
    with pytest.raises(probes.NoSpelling, match="no work-graph node stage"):
        probes.build(REMAINING_RECURSION, "crossgl")
    with pytest.raises(probes.ProbeError, match="no node-stage probe for glsl"):
        probes.build(REMAINING_RECURSION, "glsl")


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
def test_node_probes_are_valid_and_a_library_needs_an_entry_of_the_stage():
    for baseline in (False, True):
        probe = probes.build(REMAINING_RECURSION, "hlsl", baseline=baseline)
        verdict = validators.validate("hlsl", probe.source, "node", probe.entry)
        assert verdict.accepted, verdict.log
    # DXC compiles a library without entry points, which no pipeline can run.
    plain = probe.source.replace('[Shader("node")]\n', "")
    verdict = validators.validate("hlsl", plain, "node", probe.entry)
    assert not verdict.accepted
    assert "no node entry point" in verdict.log


MAKE_NOP = Feature(
    "hlsl",
    "method",
    "DxHitObject.MakeNop",
    Signature(Type.other("DxHitObject"), ()),
    group="DxHitObject",
    crossgl="MakeNop",
    spelling="dx::HitObject::MakeNop",
    stage="ray-generation",
)


def test_a_static_method_returning_a_hit_object_keeps_it_in_a_local():
    hlsl = probes.build(MAKE_NOP, "hlsl").source
    assert "    dx::HitObject r = dx::HitObject::MakeNop();" in hlsl
    crossgl = probes.build(MAKE_NOP, "crossgl").source
    assert "            dx::HitObject r = obj.MakeNop();" in crossgl
    if validators.available("hlsl"):
        verdict = validators.validate("hlsl", hlsl, "ray-generation")
        assert verdict.accepted, verdict.log


def node_method(owner, name, result, *params):
    return Feature(
        "hlsl",
        "method",
        f"{owner}.{name}",
        Signature(result, tuple(Param(p) for p in params)),
        group=owner,
        crossgl=name,
        spelling=name,
        stage="node",
    )


NODE_METHODS = [
    node_method("NodeOutput", "IsValid", Type("bool")),
    node_method(
        "NodeOutput",
        "GetThreadNodeOutputRecords",
        Type.other("ThreadNodeOutputRecords"),
        Type("uint"),
    ),
    node_method("EmptyNodeOutput", "ThreadIncrementOutputCount", VOID, Type("uint")),
    node_method("EmptyNodeInput", "Count", Type("uint")),
    node_method("GroupNodeInputRecords", "Count", Type("uint")),
    node_method("RWDispatchNodeInputRecord", "FinishedCrossGroupSharing", Type("bool")),
    node_method("GroupOrThreadNodeOutputRecords", "OutputComplete", VOID),
]


def test_node_objects_are_entry_parameters_of_their_kind_of_node():
    is_valid, records, _, empty_input, _, shared, complete = (
        probes.build(f, "hlsl").source for f in NODE_METHODS
    )
    assert "void main([MaxRecords(1)] NodeOutput<Record> obj) {" in is_valid
    assert '[NodeLaunch("broadcasting")]\n[NodeDispatchGrid(1, 1, 1)]' in is_valid
    # A call whose records object is dropped crashes DXC, so the probe keeps it.
    assert (
        "    ThreadNodeOutputRecords<Record> r = obj.GetThreadNodeOutputRecords(a0);"
        in (records)
    )
    # An input record makes the node a coalescing one, which has no fixed grid.
    assert '[NodeLaunch("coalescing")]\n[NumThreads(1, 1, 1)]' in empty_input
    assert "void main([MaxRecords(1)] EmptyNodeInput obj) {" in empty_input
    assert "struct [NodeTrackRWInputSharing] Record {" in shared
    assert "globallycoherent RWDispatchNodeInputRecord<Record> obj" in shared
    assert "void main([MaxRecords(1)] NodeOutput<Record> output) {" in complete
    assert (
        "    ThreadNodeOutputRecords<Record> obj = output.GetThreadNodeOutputRecords(1);"
        in (complete)
    )
    for method in NODE_METHODS:
        with pytest.raises(probes.NoSpelling, match="no work-graph node stage"):
            probes.build(method, "crossgl")


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
@pytest.mark.parametrize("node_feature", NODE_METHODS, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_node_object_probes_are_valid(node_feature, baseline):
    probe = probes.build(node_feature, "hlsl", baseline=baseline)
    verdict = validators.validate("hlsl", probe.source, "node", probe.entry)
    assert verdict.accepted, verdict.log


EMIT_STREAM_VERTEX = feature(
    "EmitStreamVertex", VOID, Param(Type("int"), "const"), stage="geometry"
)


def test_geometry_features_get_a_point_geometry_shader():
    glsl = probes.build(EMIT_STREAM_VERTEX, "glsl")
    assert glsl.suffix == ".geom"
    assert "layout(points) in;\nlayout(points, max_vertices = 1) out;" in glsl.source
    assert "    EmitStreamVertex(0);" in glsl.source
    assert "local_size_x" not in glsl.source
    crossgl = probes.build(EMIT_STREAM_VERTEX, "crossgl").source
    assert "    geometry {\n        layout(points) in;" in crossgl
    assert "            EmitStreamVertex(0);" in crossgl
    assert "PointStream" not in crossgl


STREAM_APPEND = Feature(
    "hlsl",
    "method",
    "Stream.Append",
    Signature(VOID, (Param(Type.other("Vertex")),)),
    group="Stream",
    crossgl="Append",
    spelling="Append",
    stage="geometry",
)
STREAM_RESTART = replace(
    STREAM_APPEND,
    name="Stream.RestartStrip",
    signature=Signature(VOID, ()),
    crossgl="RestartStrip",
    spelling="RestartStrip",
)


def test_stream_methods_are_called_on_the_geometry_shaders_output_stream():
    hlsl = probes.build(STREAM_APPEND, "hlsl").source
    assert (
        "void main(point float4 position[1] : POSITION, inout PointStream<Vertex> obj) {"
        in hlsl
    )
    assert "    emitted.value = position[0];\n    obj.Append(emitted);" in hlsl
    crossgl = probes.build(STREAM_APPEND, "crossgl").source
    assert "    struct Vertex {\n        vec4 value @ TEXCOORD0;\n    }" in crossgl
    assert (
        "void main(point vec4 position[1] @ POSITION, inout PointStream<Vertex> obj)"
        in crossgl
    )
    assert "            obj.Append(emitted);" in crossgl
    assert "layout(points)" not in crossgl
    # A stream's methods need the stream the entry point takes; nothing else declares it.
    assert "PointStream<Vertex> obj;" not in crossgl


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
@pytest.mark.parametrize(
    "probe_feature", [STREAM_APPEND, STREAM_RESTART], ids=lambda f: f.name
)
def test_stream_probes_are_valid_hlsl(probe_feature):
    for baseline in (False, True):
        probe = probes.build(probe_feature, "hlsl", baseline=baseline)
        verdict = validators.validate("hlsl", probe.source, "geometry", probe.entry)
        assert verdict.accepted, verdict.log


def test_geometry_probes_are_valid():
    if not validators.available("glsl"):
        pytest.skip("no glsl validator")
    for name, params in (
        ("EmitVertex", ()),
        ("EmitStreamVertex", EMIT_STREAM_VERTEX.signature.params),
    ):
        spelled = feature(name, VOID, *params, stage="geometry")
        for baseline in (False, True):
            probe = probes.build(spelled, "glsl", baseline=baseline)
            verdict = validators.validate("glsl", probe.source, "geometry", probe.entry)
            assert verdict.accepted, verdict.log


UINT = Type("uint")
VEC4 = Type("float", (4,))
VEC2 = Type("float", (2,))
STAGE_FUNCTIONS = {
    "tessellation-control": (
        "Process2DQuadTessFactorsAvg",
        VOID,
        VEC4,
        VEC2,
        Param(VEC4, "out"),
        Param(VEC2, "out"),
        Param(VEC2, "out"),
    ),
    "mesh": ("SetMeshOutputCounts", VOID, UINT, UINT),
    "task": ("DispatchMesh", VOID, UINT, UINT, UINT, Type("", (), "udt")),
}


def stage_function(language, stage):
    name, result, *params = STAGE_FUNCTIONS[stage]
    signature = Signature(
        result, tuple(p if isinstance(p, Param) else Param(p) for p in params)
    )
    return Feature(language, "function", name, signature, crossgl=name, stage=stage)


def test_tessellation_factor_functions_run_in_the_patch_constant_function():
    feature = stage_function("hlsl", "tessellation-control")
    hlsl = probes.build(feature, "hlsl").source
    constants, entry = hlsl.split("[domain(")
    assert "PatchConstants constants(InputPatch<ControlPoint, 3> patch) {" in constants
    assert "    Process2DQuadTessFactorsAvg(a0, a1, a2, a3, a4);" in constants
    assert '[patchconstantfunc("constants")]' in entry
    assert "numthreads" not in hlsl
    slang = probes.build(replace(feature, language="slang"), "slang").source
    assert '[shader("hull")]\n[domain("tri")]' in slang
    crossgl = probes.build(feature, "crossgl").source
    assert "    tessellation_control {\n        buffer float* fin;" in crossgl
    assert "            Process2DQuadTessFactorsAvg(a0, a1, a2, a3, a4);" in crossgl
    assert "            @patchconstantfunc(constants) {" in crossgl


def test_mesh_functions_run_before_the_mesh_shader_writes_its_triangle():
    hlsl = probes.build(stage_function("hlsl", "mesh"), "hlsl").source
    call = hlsl.index("    SetMeshOutputCounts(a0, a1);")
    assert hlsl.index("    tris[0] = uint3(0, 1, 2);") > call
    assert '[numthreads(1, 1, 1)]\n[outputtopology("triangle")]' in hlsl
    crossgl = probes.build(stage_function("hlsl", "mesh"), "crossgl").source
    assert "    mesh {\n        buffer uint* uin;" in crossgl
    assert "            SetMeshOutputCounts(a0, a1);" in crossgl


@pytest.mark.parametrize("language", ["hlsl", "slang", "crossgl"])
def test_a_task_shader_hands_its_payload_to_dispatch_mesh(language):
    source = probes.build(stage_function("hlsl", "task"), language).source
    assert "DispatchMesh(a0, a1, a2, payload);" in source
    declaration = {
        "crossgl": "groupshared Payload payload @ taskPayloadSharedEXT;",
        "hlsl": "groupshared Payload payload;",
        "slang": "groupshared Payload payload;",
    }
    assert declaration[language] in source
    # Slang's DispatchMesh takes the payload by reference.
    by_reference = stage_function("slang", "task")
    *values, _ = by_reference.signature.params
    by_reference = replace(
        by_reference,
        signature=Signature(VOID, (*values, Param(Type("int"), "inout"))),
    )
    assert (
        "DispatchMesh(a0, a1, a2, payload);" in probes.build(
            by_reference, language
        ).source
    )


def trace_ray(language):
    """TraceRay as the HLSL list declares it, or as the Slang list does with its generic payload read as ``int``."""
    hlsl = language == "hlsl"
    acceleration = Type(
        "", (), "acceleration_struct" if hlsl else "RaytracingAccelerationStructure"
    )
    ray = Type("", (), "ray_desc" if hlsl else "RayDesc")
    payload = Type("", (), "udt") if hlsl else Type("int")
    params = (acceleration, *[UINT] * 5, ray, Param(payload, "inout"))
    signature = Signature(
        VOID, tuple(p if isinstance(p, Param) else Param(p) for p in params)
    )
    return Feature(
        language,
        "function",
        "TraceRay",
        signature,
        crossgl="TraceRay",
        stage="closest-hit",
    )


@pytest.mark.parametrize(
    ("language", "acceleration", "origin"),
    [
        (
            "hlsl",
            "RaytracingAccelerationStructure u0 : register(t8);",
            "float3(fin[i + 0u], fin[i + 1u], fin[i + 2u])",
        ),
        (
            "slang",
            "RaytracingAccelerationStructure u0;",
            "float3(fin[i + 0u], fin[i + 1u], fin[i + 2u])",
        ),
        (
            "crossgl",
            "uniform accelerationStructureEXT u0;",
            "vec3(fin[i + 0u], fin[i + 1u], fin[i + 2u])",
        ),
    ],
)
def test_a_hit_shader_traces_a_ray_with_its_payload(language, acceleration, origin):
    native = "slang" if language == "slang" else "hlsl"
    source = probes.build(trace_ray(native), language).source
    assert acceleration in source
    assert "RayDesc a6;" in source
    assert f"a6.Origin = {origin};" in source
    assert "a6.TMax = 1.0;" in source
    assert "TraceRay(u0, a1, a2, a3, a4, a5, a6, payload);" in source


@pytest.mark.parametrize("language", ["hlsl", "slang"])
def test_trace_ray_probes_are_valid(language):
    if not validators.available(language):
        pytest.skip(f"no {language} validator")
    for baseline in (False, True):
        probe = probes.build(trace_ray(language), language, baseline=baseline)
        verdict = validators.validate(
            language, probe.source, "closest-hit", probe.entry
        )
        assert verdict.accepted, verdict.log


UDT = Type("", (), "udt")
REPORT_HIT = Feature(
    "hlsl",
    "function",
    "ReportHit",
    Signature(Type("bool"), (Param(Type("float")), Param(UINT), Param(UDT))),
    crossgl="ReportHit",
    stage="intersection",
)
GET_ATTRIBUTES = Feature(
    "hlsl",
    "method",
    "DxHitObject.GetAttributes",
    Signature(VOID, (Param(UDT, "out"),)),
    group="DxHitObject",
    crossgl="GetAttributes",
    spelling="GetAttributes",
    stage="ray-generation",
)


@pytest.mark.parametrize(
    ("language", "barycentrics"),
    [
        ("hlsl", "float2(fin[i + 1u], fin[i + 2u])"),
        ("crossgl", "vec2(fin[i + 1u], fin[i + 2u])"),
    ],
)
def test_an_intersection_shader_reports_triangle_attributes(language, barycentrics):
    source = probes.build(REPORT_HIT, language).source
    assert "BuiltInTriangleIntersectionAttributes a2;" in source
    assert f"a2.barycentrics = {barycentrics};" in source
    assert "bool r = ReportHit(a0, a1, a2);" in source


def test_a_hit_object_writes_its_attributes_to_a_local():
    for language in ("hlsl", "crossgl"):
        source = probes.build(GET_ATTRIBUTES, language).source
        assert "BuiltInTriangleIntersectionAttributes a0;" in source
        assert "barycentrics" not in source
        assert "obj.GetAttributes(a0);" in source


def test_a_user_defined_inout_parameter_where_no_ray_starts_stays_untested():
    swapped = Signature(Type("bool"), (Param(UDT, "inout"),))
    with pytest.raises(probes.ProbeError, match="parameters of type udt"):
        probes.build(replace(REPORT_HIT, signature=swapped), "hlsl")


def hit_object(name, result, *params):
    return Feature(
        "hlsl",
        "method",
        f"DxHitObject.{name}",
        Signature(result, params),
        group="DxHitObject",
        crossgl=name,
        spelling=f"dx::HitObject::{name}",
        stage="ray-generation",
    )


HIT = Type.other("DxHitObject")
INVOKE = hit_object("Invoke", VOID, Param(HIT), Param(UDT, "inout"))
TRACE_FROM_HIT_OBJECT = hit_object(
    "TraceRay",
    HIT,
    *(Param(t) for t in (Type.other("acceleration_struct"), *[UINT] * 5)),
    Param(Type.other("ray_desc")),
    Param(UDT, "inout"),
)
FROM_RAY_QUERY = hit_object("FromRayQuery", HIT, Param(Type.other("RayQuery")))


@pytest.mark.parametrize(
    ("language", "struct", "call"),
    [
        ("hlsl", "struct [raypayload] Payload", "dx::HitObject::Invoke(a0, payload);"),
        ("crossgl", "    struct Payload {", "obj.Invoke(a0, payload);"),
    ],
)
def test_a_ray_generation_shader_passes_a_local_payload(language, struct, call):
    source = probes.build(INVOKE, language).source
    assert struct in source
    assert "Payload payload;" in source
    assert "dx::HitObject a0;" in source
    assert call in source
    traced = probes.build(TRACE_FROM_HIT_OBJECT, language).source
    assert "Payload payload;" in traced
    assert "TraceRay(u0, a1, a2, a3, a4, a5, a6, payload);" in traced


def test_a_ray_generation_shader_declares_a_payload_only_when_it_traces():
    for language in ("hlsl", "crossgl"):
        source = probes.build(FROM_RAY_QUERY, language).source
        assert "Payload" not in source


def test_a_hit_object_is_made_from_a_ray_query_that_has_run():
    for language in ("hlsl", "crossgl"):
        source = probes.build(FROM_RAY_QUERY, language).source
        assert "RayQuery<RAY_FLAG_NONE> a0;\n" in source
        assert "a0.Proceed();\n" in source
        assert "FromRayQuery(a0);" in source


@pytest.mark.parametrize(
    "feature", [INVOKE, TRACE_FROM_HIT_OBJECT, FROM_RAY_QUERY], ids=lambda f: f.name
)
def test_hit_object_constructor_probes_are_valid(feature):
    if not validators.available("hlsl"):
        pytest.skip("no hlsl validator")
    for baseline in (False, True):
        probe = probes.build(feature, "hlsl", baseline=baseline)
        verdict = validators.validate("hlsl", probe.source, feature.stage, probe.entry)
        assert verdict.accepted, verdict.log


@pytest.mark.parametrize("feature", [REPORT_HIT, GET_ATTRIBUTES])
def test_hit_attribute_probes_are_valid(feature):
    if not validators.available("hlsl"):
        pytest.skip("no hlsl validator")
    for baseline in (False, True):
        probe = probes.build(feature, "hlsl", baseline=baseline)
        verdict = validators.validate("hlsl", probe.source, feature.stage, probe.entry)
        assert verdict.accepted, verdict.log


def test_stages_without_a_function_probe_say_so():
    for stage in STAGE_FUNCTIONS:
        with pytest.raises(probes.ProbeError, match=f"no {stage}-stage probe for glsl"):
            probes.build(stage_function("glsl", stage), "glsl")


@pytest.mark.parametrize("language", ["hlsl", "slang"])
@pytest.mark.parametrize("stage", list(STAGE_FUNCTIONS))
def test_stage_function_probes_are_valid(language, stage):
    if not validators.available(language):
        pytest.skip(f"no {language} validator")
    feature = stage_function(language, stage)
    if (language, stage) == ("slang", "tessellation-control"):
        # Slang has the tessellation-factor functions only for HLSL targets; any function checks the hull shader.
        scalar = Type("float")
        feature = replace(
            feature, name="abs", signature=Signature(scalar, (Param(scalar),))
        )
    probe = probes.build(feature, language)
    verdict = validators.validate(language, probe.source, stage, probe.entry)
    assert verdict.accepted, verdict.log
