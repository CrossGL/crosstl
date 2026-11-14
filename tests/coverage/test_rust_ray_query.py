"""spirv_std's RayQuery and AccelerationStructure methods in the Rust feature list, and their probes."""

import pytest

from tools.coverage import probes, rustgpu, validators
from tools.coverage.features import ray_query

# An excerpt of spirv-std 0.11's ray_tracing.rs, with the shapes the parser reads: asm! strings that continue on the
# next line after a backslash, an intersection operand defined by an OpConstant, a method without a receiver, a
# generic payload, and doc comments that name other instructions.
RAY_TRACING = r"""
impl AccelerationStructure {
    /// Converts a 64-bit integer into an [`AccelerationStructure`].
    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpConvertUToAccelerationStructureKHR")]
    #[inline]
    pub unsafe fn from_u64(id: u64) -> AccelerationStructure {
        unsafe {
            let mut result_slot = core::mem::MaybeUninit::uninit();
            asm! {
                "%ret = OpTypeAccelerationStructureKHR",
                "%result = OpConvertUToAccelerationStructureKHR %ret {id}",
                "OpStore {result_slot} %result",
                id = in(reg) id,
                result_slot = in(reg) result_slot.as_mut_ptr(),
            }
            result_slot.assume_init()
        }
    }

    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpTraceRayKHR")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn trace_ray<T>(
        &self,
        ray_flags: RayFlags,
        cull_mask: i32,
        sbt_offset: i32,
        sbt_stride: i32,
        miss_index: i32,
        ray_origin: Vec3,
        ray_tmin: f32,
        ray_direction: Vec3,
        ray_tmax: f32,
        payload: &mut T,
    ) {
        unsafe {
            asm! {
                "%acceleration_structure = OpLoad _ {acceleration_structure}",
                "OpTraceRayKHR \
                %acceleration_structure \
                {ray_flags} \
                {payload}",
                acceleration_structure = in(reg) self,
                ray_flags = in(reg) ray_flags.bits(),
                payload = in(reg) payload,
            }
        }
    }
}

impl RayQuery {
    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpRayQueryInitializeKHR")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn initialize(
        &mut self,
        acceleration_structure: &AccelerationStructure,
        ray_flags: RayFlags,
        cull_mask: u32,
        ray_origin: Vec3,
        ray_tmin: f32,
        ray_direction: Vec3,
        ray_tmax: f32,
    ) {
        unsafe {
            asm! {
                "%acceleration_structure = OpLoad _ {acceleration_structure}",
                "OpRayQueryInitializeKHR \
                    {ray_query} \
                    %acceleration_structure \
                    {ray_tmax}",
                ray_query = in(reg) self,
                acceleration_structure = in(reg) acceleration_structure,
                ray_tmax = in(reg) ray_tmax,
            }
        }
    }

    /// Allow traversal to proceed. A previous call to [`Self::proceed`] must not have returned `false`.
    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpRayQueryProceedKHR")]
    #[inline]
    pub unsafe fn proceed(&self) -> bool {
        unsafe {
            let mut result = false;
            asm! {
                "%bool = OpTypeBool",
                "%result = OpRayQueryProceedKHR %bool {ray_query}",
                "OpStore {result} %result",
                ray_query = in(reg) self,
                result = in(reg) &mut result,
            }
            result
        }
    }

    /// Unlike `OpRayQueryGetIntersectionTypeKHR` for the committed intersection, this reads the candidate.
    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpRayQueryGetIntersectionTypeKHR")]
    #[inline]
    pub unsafe fn get_candidate_intersection_type(&self) -> CandidateIntersection {
        unsafe {
            let result: u32;
            asm! {
                "%u32 = OpTypeInt 32 0",
                "%intersection = OpConstant %u32 0",
                "{result} = OpRayQueryGetIntersectionTypeKHR %u32 {ray_query} %intersection",
                ray_query = in(reg) self,
                result = out(reg) result,
            }
            match result {
                0 => CandidateIntersection::Triangle,
                _ => CandidateIntersection::AABB,
            }
        }
    }

    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpRayQueryGetIntersectionTKHR")]
    #[inline]
    pub unsafe fn get_committed_intersection_t(&self) -> f32 {
        unsafe {
            let result;
            asm! {
                "%u32 = OpTypeInt 32 0",
                "%intersection = OpConstant %u32 1",
                "{result} = OpRayQueryGetIntersectionTKHR typeof{result} {ray_query} %intersection",
                ray_query = in(reg) self,
                result = out(reg) result,
            }
            result
        }
    }

    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpRayQueryGetIntersectionCandidateAABBOpaqueKHR")]
    #[inline]
    pub unsafe fn get_intersection_candidate_aabb_opaque(&self) -> bool {
        unsafe {
            let mut result = false;
            asm! {
                "%bool = OpTypeBool",
                "%result = OpRayQueryGetIntersectionCandidateAABBOpaqueKHR %bool {ray_query}",
                "OpStore {result} %result",
                ray_query = in(reg) self,
                result = in(reg) &mut result,
            }
            result
        }
    }

    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpRayQueryGetIntersectionObjectToWorldKHR")]
    #[inline]
    pub unsafe fn get_committed_intersection_object_to_world(&self) -> Matrix4x3 {
        unsafe {
            let mut result = Default::default();
            asm! {
                "%u32 = OpTypeInt 32 0",
                "%intersection = OpConstant %u32 1",
                "%result = OpRayQueryGetIntersectionObjectToWorldKHR typeof*{result} {ray_query} %intersection",
                "OpStore {result} %result",
                ray_query = in(reg) self,
                result = in(reg) &mut result,
            }
            result
        }
    }

    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpRayQueryGetIntersectionTriangleVertexPositionsKHR")]
    #[inline]
    pub unsafe fn get_intersection_triangle_vertex_positions(&self) -> [Vec3; 3] {
        unsafe {
            let mut result = Default::default();
            asm! {
                "%u32 = OpTypeInt 32 0",
                "%intersection = OpConstant %u32 0",
                "%result = OpRayQueryGetIntersectionTriangleVertexPositionsKHR typeof*{result} {ray_query} %intersection",
                "OpStore {result} %result",
                ray_query = in(reg) self,
                result = in(reg) &mut result,
            }
            result
        }
    }
}
"""


@pytest.fixture(scope="module")
def features():
    return {feature.name: feature for feature in ray_query.parse(RAY_TRACING)}


def test_ray_tracing_methods_are_named_by_their_type(features):
    assert sorted(features) == [
        "AccelerationStructure::from_u64",
        "AccelerationStructure::trace_ray",
        "RayQuery::get_candidate_intersection_type",
        "RayQuery::get_committed_intersection_object_to_world",
        "RayQuery::get_committed_intersection_t",
        "RayQuery::get_intersection_candidate_aabb_opaque",
        "RayQuery::get_intersection_triangle_vertex_positions",
        "RayQuery::initialize",
        "RayQuery::proceed",
    ]
    assert {feature.group for feature in features.values()} == {
        "AccelerationStructure",
        "RayQuery",
    }


def test_each_method_is_the_dxr_method_of_its_instruction_and_intersection(features):
    crossgl = {name: feature.crossgl for name, feature in features.items()}
    assert crossgl == {
        "AccelerationStructure::from_u64": "",
        "AccelerationStructure::trace_ray": "TraceRay",
        "RayQuery::get_candidate_intersection_type": "{}.CandidateType()",
        "RayQuery::get_committed_intersection_object_to_world": (
            "{}.CommittedObjectToWorld4x3()"
        ),
        "RayQuery::get_committed_intersection_t": "{}.CommittedRayT()",
        # DXR asks whether the candidate is non-opaque.
        "RayQuery::get_intersection_candidate_aabb_opaque": (
            "!{}.CandidateProceduralPrimitiveNonOpaque()"
        ),
        "RayQuery::get_intersection_triangle_vertex_positions": (
            "{}.CandidateTriangleVertexPositions({})"
        ),
        "RayQuery::initialize": "{}.TraceRayInline({}, {}, {}, {})",
        "RayQuery::proceed": "{}.Proceed()",
    }
    assert (
        "makes none from an address" in features["AccelerationStructure::from_u64"].note
    )
    with pytest.raises(probes.NoSpelling):
        probes.build(features["AccelerationStructure::from_u64"], "crossgl")


def test_methods_are_called_on_their_receiver_and_a_constructor_by_its_path(features):
    assert features["RayQuery::proceed"].spelling == "unsafe .proceed"
    assert features["AccelerationStructure::from_u64"].spelling == (
        "unsafe spirv_std::ray_tracing::AccelerationStructure::from_u64"
    )
    trace = features["AccelerationStructure::trace_ray"]
    assert trace.stage == "closest-hit"
    assert trace.signature.params[0].type.opaque == "AccelerationStructure"
    assert trace.signature.params[-1].type.opaque == "udt"
    assert trace.signature.params[-1].direction == "inout"


def test_ray_query_probes_make_the_query_with_spirv_std_macro(features):
    native = probes.build(features["RayQuery::get_candidate_intersection_type"], "rust")
    source = native.source
    assert source.startswith("#![no_std]\n#![feature(asm_experimental_arch)]\n")
    assert "use spirv_std::ray_tracing::RayQuery;" in source
    assert "    spirv_std::ray_query!(let mut a0);" in source
    assert (
        "let r: u32 = (unsafe { a0.get_candidate_intersection_type() }) as u32;"
        in source
    )
    crossgl = probes.build(
        features["RayQuery::get_candidate_intersection_type"], "crossgl"
    ).source
    assert "RayQuery<RAY_FLAG_NONE> a0;" in crossgl
    assert "uint r = a0.CandidateType();" in crossgl


def test_ray_query_results_no_buffer_holds_are_read_as_values(features):
    world = probes.build(
        features["RayQuery::get_committed_intersection_object_to_world"], "rust"
    ).source
    assert (
        "let r: Vec3 = (unsafe { a0.get_committed_intersection_object_to_world() }).w_axis;"
        in world
    )
    positions = probes.build(
        features["RayQuery::get_intersection_triangle_vertex_positions"], "rust"
    ).source
    assert (
        "let r: Vec3 = (unsafe { a0.get_intersection_triangle_vertex_positions() })[0];"
        in positions
    )
    crossgl = probes.build(
        features["RayQuery::get_intersection_triangle_vertex_positions"], "crossgl"
    ).source
    assert "vec3 a1[3];\n" in crossgl
    assert "a0.CandidateTriangleVertexPositions(a1);" in crossgl


def test_initialize_takes_an_acceleration_structure_and_a_ray(features):
    initialize = features["RayQuery::initialize"]
    native = probes.build(initialize, "rust").source
    assert "use spirv_std::ray_tracing::AccelerationStructure;" in native
    assert (
        "#[spirv(descriptor_set = 0, binding = 8)] u0: &AccelerationStructure,"
        in native
    )
    assert (
        "unsafe { a0.initialize(u0, spirv_std::ray_tracing::RayFlags::from_bits_retain(a2), a3, a4, a5, a6, a7) };"
        in native
    )
    crossgl = probes.build(initialize, "crossgl").source
    assert "uniform accelerationStructureEXT u0;" in crossgl
    assert "a4.Origin = " in crossgl
    assert "a0.TraceRayInline(u0, a2, a3, a4);" in crossgl


def test_trace_ray_passes_the_closest_hit_shader_payload(features):
    trace = features["AccelerationStructure::trace_ray"]
    native = probes.build(trace, "rust").source
    assert "#[spirv(closest_hit)]" in native
    assert "#[spirv(incoming_ray_payload)] payload: &mut Payload," in native
    assert (
        "u0.trace_ray(spirv_std::ray_tracing::RayFlags::from_bits_retain(a1)," in native
    )
    assert native.rstrip().endswith("a9, payload) };\n    outp[i] = 0;\n}")
    crossgl = probes.build(trace, "crossgl").source
    assert "TraceRay(u0, a1, a2, a3, a4, a5, a6, payload);" in crossgl


def test_rust_gpu_enables_ray_queries_and_their_vertex_positions():
    assert {"RayQueryKHR", "RayQueryPositionFetchKHR"} <= set(rustgpu.CAPABILITIES)
    assert {"SPV_KHR_ray_query", "SPV_KHR_ray_tracing_position_fetch"} <= set(
        rustgpu.EXTENSIONS
    )


@pytest.mark.skipif(
    not validators.available("rust"), reason="rust-gpu is not installed"
)
def test_ray_tracing_probes_compile_with_rust_gpu(features):
    for feature in features.values():
        probe = probes.build(feature, "rust")
        verdict = validators.validate(
            "rust", probe.source, feature.stage, entry=probe.entry
        )
        assert verdict.accepted, (feature.name, verdict.log)
