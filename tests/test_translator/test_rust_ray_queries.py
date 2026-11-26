"""DXR inline ray queries in the rust-gpu output (:mod:`crosstl.translator.codegen.rust_gpu_ray_queries`).

A ``RayQuery<FLAGS>`` local is spirv-std's ``ray_query!`` variable, each CrossGL method the spirv-std method that issues
the same SPV_KHR_ray_query instruction, and an acceleration structure an ``&AccelerationStructure`` binding. Each
program is translated as ``crosstl.translate`` translates it, and each generated crate compiles with rust-gpu when it
is built.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_codegen.test_rust_codegen import assert_generated_rust_smoke_compiles


def _kernel(body, helpers="", declarations="uniform accelerationStructureEXT scene;"):
    return f"""
shader Probe {{
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        {declarations}
        {helpers}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def _translate(source, tmp_path, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="rust", format_output=False)


def _flat(code):
    return " ".join(code.split())


TRACE = """
            RayDesc ray;
            ray.Origin = vec3(fin[i], 0.0, 0.0);
            ray.TMin = 0.001;
            ray.Direction = vec3(0.0, 0.0, 1.0);
            ray.TMax = 100.0;
            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFFu, ray);
"""


def test_a_query_traces_a_ray_desc_against_a_bound_acceleration_structure(tmp_path):
    code = _translate(
        _kernel(TRACE + """
            while (q.Proceed()) {
                if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                    q.CommitNonOpaqueTriangleHit();
                }
            }
            float t = -1.0;
            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                t = q.CommittedRayT();
            }
            outp[i] = t;
"""),
        tmp_path,
    )
    flat = _flat(code)
    assert "use spirv_std::ray_tracing::{AccelerationStructure, RayQuery};" in code
    assert "#![feature(asm_experimental_arch)]" in code
    assert (
        "#[spirv(descriptor_set = 0, binding = 2)] scene: &AccelerationStructure"
        in flat
    )
    # DXR's RayDesc becomes a struct with its four fields, and TraceRayInline passes them to initialize.
    assert (
        "pub struct RayDesc { pub Origin: Vec3, pub TMin: f32, pub Direction: Vec3, pub TMax: f32,"
        in flat
    )
    assert "spirv_std::ray_query!(let mut q);" in flat
    assert (
        "unsafe { q.initialize(scene, spirv_std::ray_tracing::RayFlags::from_bits_retain(0x0u32), 255, "
        "ray.Origin, ray.TMin, ray.Direction, ray.TMax) }" in flat
    )
    assert "while unsafe { q.proceed() }" in flat
    # The intersection kinds are spirv-std enums with the DXR discriminants.
    assert "(unsafe { q.get_candidate_intersection_type() }) as u32 == 0x0u32" in flat
    assert "unsafe { q.confirm_intersection() };" in flat
    assert "(unsafe { q.get_committed_intersection_type() }) as u32 == 0x1u32" in flat
    assert "t = unsafe { q.get_committed_intersection_t() };" in flat
    assert_generated_rust_smoke_compiles(code, tmp_path)


GETTERS = [
    ("float", "q.RayTMin()", "unsafe { q.get_ray_t_min() }"),
    ("uint", "q.RayFlags()", "(unsafe { q.get_ray_flags() }).bits()"),
    ("vec3", "q.WorldRayOrigin()", "unsafe { q.get_world_ray_origin() }"),
    ("vec3", "q.WorldRayDirection()", "unsafe { q.get_world_ray_direction() }"),
    (
        "float",
        "q.CandidateTriangleRayT()",
        "unsafe { q.get_candidate_intersection_t() }",
    ),
    (
        "uint",
        "q.CandidateInstanceID()",
        "unsafe { q.get_candidate_intersection_instance_custom_index() }",
    ),
    (
        "uint",
        "q.CommittedInstanceIndex()",
        "unsafe { q.get_committed_intersection_instance_id() }",
    ),
    (
        "uint",
        "q.CommittedInstanceContributionToHitGroupIndex()",
        "unsafe { q.get_committed_intersection_shader_binding_table_record_offset() }",
    ),
    (
        "uint",
        "q.CandidateGeometryIndex()",
        "unsafe { q.get_candidate_intersection_geometry_index() }",
    ),
    (
        "uint",
        "q.CommittedPrimitiveIndex()",
        "unsafe { q.get_committed_intersection_primitive_index() }",
    ),
    (
        "vec2",
        "q.CandidateTriangleBarycentrics()",
        "unsafe { q.get_candidate_intersection_barycentrics() }",
    ),
    (
        "bool",
        "q.CommittedTriangleFrontFace()",
        "unsafe { q.get_committed_intersection_front_face() }",
    ),
    # DXR asks whether a candidate is non-opaque, SPIR-V whether it is opaque.
    (
        "bool",
        "q.CandidateProceduralPrimitiveNonOpaque()",
        "!(unsafe { q.get_intersection_candidate_aabb_opaque() })",
    ),
    (
        "vec3",
        "q.CommittedObjectRayOrigin()",
        "unsafe { q.get_committed_intersection_object_ray_origin() }",
    ),
    (
        "vec3",
        "q.CandidateObjectRayDirection()",
        "unsafe { q.get_candidate_intersection_object_ray_direction() }",
    ),
]


def test_each_getter_is_the_spirv_std_method_of_its_instruction(tmp_path):
    reads = []
    for index, (crossgl_type, call, _rust) in enumerate(GETTERS):
        reads.append(f"{crossgl_type} r{index} = {call};")
    body = TRACE + "\n".join(reads) + "\noutp[i] = r0;"
    code = _translate(_kernel(body), tmp_path)
    flat = _flat(code)
    for _crossgl_type, _call, rust in GETTERS:
        assert rust in flat, rust
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_template_flags_and_glsl_constants_join_the_initialize_flags(tmp_path):
    # The GLSL importer writes rayQueryEXT, the seven-operand TraceRayInline and the GLSL constants.
    code = _translate(
        _kernel("""
            RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_TRIANGLES> q;
            q.TraceRayInline(scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFFu, vec3(fin[i], 0.0, 0.0), 0.01,
                             vec3(0.0, 0.0, 1.0), 100.0);
            q.Proceed();
            rayQueryEXT g;
            g.TraceRayInline(scene, gl_RayFlagsOpaqueEXT, 0xFF, vec3(0.0), 0.0, vec3(0.0, 0.0, 1.0), 1.0);
            g.Proceed();
            outp[i] = (g.CommittedStatus() == gl_RayQueryCommittedIntersectionTriangleEXT) ? q.CommittedRayT() : 0.0;
"""),
        tmp_path,
    )
    flat = _flat(code)
    # DXR ORs the query type's flags into TraceRayInline's (D3D12 raytracing specification, "RayQuery").
    assert (
        "RayFlags::from_bits_retain((0x10u32) | 0x101u32), 255, Vec3::new(fin[i as usize], 0.0, 0.0), 0.01,"
        in flat
    )
    assert "spirv_std::ray_query!(let mut g);" in flat
    assert (
        "RayFlags::from_bits_retain(0x1u32), 255, Vec3::new(0.0, 0.0, 0.0), 0.0,"
        in flat
    )
    assert "(unsafe { g.get_committed_intersection_type() }) as u32 == 0x1u32" in flat
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_generated_hits_and_vertex_positions_without_a_method_are_inline_spirv(
    tmp_path,
):
    code = _translate(
        _kernel(TRACE + """
            if (q.Proceed() && q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                q.CommitProceduralPrimitiveHit(0.5);
            }
            vec3 candidate[3];
            vec3 committed[3];
            q.CandidateTriangleVertexPositions(candidate);
            q.CommittedTriangleVertexPositions(committed);
            outp[i] = candidate[0].x + committed[2].z;
"""),
        tmp_path,
    )
    flat = _flat(code)
    assert "let mut candidate: [Vec3; 3]" in flat
    assert (
        "candidate = unsafe { q.get_intersection_triangle_vertex_positions() };" in flat
    )
    # spirv-std 0.11 has no method for OpRayQueryGenerateIntersectionKHR or the committed triangle's positions.
    assert (
        '{ let hit_t: f32 = 0.5; unsafe { asm!("OpRayQueryGenerateIntersectionKHR {} {}", in(reg) &*q, '
        "in(reg) hit_t) } }" in flat
    )
    assert '"%intersection = OpConstant %u32 1"' in flat
    assert (
        "OpRayQueryGetIntersectionTriangleVertexPositionsKHR typeof*{result} {ray_query}"
        in flat
    )
    assert "use core::arch::asm;" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_a_helper_receives_the_acceleration_structure_by_reference(tmp_path):
    code = _translate(
        _kernel(
            "outp[i] = closest(scene, vec3(fin[i], 0.0, 0.0));",
            helpers="""
        float closest(accelerationStructureEXT acc, vec3 origin) {
            RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
            q.TraceRayInline(acc, 0u, 0xFFu, origin, 0.0, vec3(0.0, 0.0, 1.0), 10.0);
            q.Proceed();
            return q.CommittedRayT();
        }
""",
        ),
        tmp_path,
    )
    flat = _flat(code)
    assert "pub fn closest(acc: &AccelerationStructure, origin: Vec3) -> f32" in flat
    assert "closest(scene, Vec3::new(fin[i as usize], 0.0, 0.0))" in flat
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_hlsl_ray_queries_translate_to_rust(tmp_path):
    source = """
RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<float> results : register(u0);

[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    RayDesc ray;
    ray.Origin = float3(id.x, 0.0, 0.0);
    ray.TMin = 0.001;
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMax = 1000.0;
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    float t = -1.0;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        t = q.CommittedRayT() + q.CommittedTriangleBarycentrics().x;
    }
    results[id.x] = t;
}
"""
    code = _translate(source, tmp_path, name="query.hlsl")
    flat = _flat(code)
    assert "scene: &AccelerationStructure" in flat
    assert "RayFlags::from_bits_retain((0x0u32) | 0x280u32)" in flat
    assert_generated_rust_smoke_compiles(code, tmp_path)


REFUSALS = [
    pytest.param(
        "RayQuery<RAY_FLAG_NONE> q; mat3x4 m = q.CommittedObjectToWorld4x3(); outp[i] = m[0].x;",
        "",
        r"ray query: 'CommittedObjectToWorld4x3' returns a 4x3 or 3x4 matrix, and glam has no non-square",
        id="object-to-world",
    ),
    pytest.param(
        "RayQuery<RAY_FLAG_NONE> q; q.TraceRayInline(scene, 0u, 0xFFu, 1.0); outp[i] = 0.0;",
        "",
        r"ray query: TraceRayInline takes a RayDesc, got 'float'",
        id="trace-without-ray-desc",
    ),
    pytest.param(
        "RayQuery<RAY_FLAG_NONE> q; q.Proceed(1u); outp[i] = 0.0;",
        "",
        r"ray query: 'Proceed' takes 0 arguments, got 1",
        id="arity",
    ),
    pytest.param(
        "RayQuery<RAY_FLAG_NONE> q; outp[i] = helper(q);",
        "float helper(RayQuery<RAY_FLAG_NONE> query) { return query.CommittedRayT(); }",
        r"ray query: parameter 'query' of 'helper' is a RayQuery",
        id="query-parameter",
    ),
]


@pytest.mark.parametrize(("body", "helpers", "reason"), REFUSALS)
def test_what_spirv_std_cannot_express_is_refused_by_name(
    body, helpers, reason, tmp_path
):
    with pytest.raises(RustLoweringRefused, match=reason):
        _translate(_kernel(body, helpers=helpers), tmp_path)


def test_a_query_outside_a_function_is_refused(tmp_path):
    source = """
struct Holder {
    RayQuery<RAY_FLAG_NONE> query;
};
RayQuery<RAY_FLAG_NONE> globalQuery;
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            outp[gl_GlobalInvocationID.x] = 0.0;
        }
    }
}
"""
    with pytest.raises(RustLoweringRefused) as refused:
        _translate(source, tmp_path)
    message = str(refused.value)
    assert "ray query: 'globalQuery' is declared at module scope" in message
    assert "ray query: member 'query' of struct 'Holder' is a RayQuery" in message
