"""Function names that need no declaration in a CrossGL program.

CrossGL's own intrinsic functions, and the built-in functions of the languages the importers read (GLSL, HLSL, MSL,
CUDA and HIP), which an importer may carry into the IR as a plain call for the generators to map, and identifiers
reserved to the implementation (:data:`RESERVED_NAME_PREFIX`). :mod:`.declared_functions` refuses a call to any other
plain name a program does not declare. Whether a target can express a built-in is left to its generator.

The GLSL, HLSL and MSL sets are the ``builtin-function`` items of the specification inventories in
``tests/conformance/inventory/data``, unqualified; ``tests/test_translator/test_undeclared_functions.py`` checks that
they stay equal. The other sets cite their sources below.
"""

from __future__ import annotations

from .. import parser as _parser
from .. import validation as _validation
from .. import wave_operations as _wave_operations
from ..codegen.cuda_library_functions import CUDA_LIBRARY_FUNCTIONS
from ..codegen.generic_function_utils import RESERVED_GENERIC_FUNCTION_BUILTIN_NAMES
from .metal_math_spellings import PI_FUNCTIONS
from .sampler_states import SAMPLER_STATE_CALL_ARGUMENTS
from .sizeof_values import SIZEOF_LAYOUTS
from .warp import OTHER_WARP_INTRINSICS, SHUFFLES

__all__ = [
    "BUILTIN_FUNCTION_NAMES",
    "RESERVED_NAME_PREFIX",
    "CROSSGL_FUNCTION_NAMES",
    "CUDA_FUNCTION_NAMES",
    "GLSL_EXTENSION_FUNCTION_NAMES",
    "GLSL_FUNCTION_NAMES",
    "HLSL_FUNCTION_NAMES",
    "METAL_FUNCTION_NAMES",
    "TYPE_TEMPLATE_NAMES",
]


def _names(text):
    return frozenset(text.split())


# GLSL 4.60 chapter 8 built-in functions (tests/conformance/inventory/data/glsl.json, category builtin-function).
GLSL_FUNCTION_NAMES = _names("""
ComputeAccessedLod EmitStreamVertex EmitVertex EndPrimitive EndStreamPrimitive abs acos acosh all allInvocations
allInvocationsEqual any anyInvocation asin asinh atan atanh atomicAdd atomicAnd atomicCompSwap atomicCounter
atomicCounterAdd atomicCounterAnd atomicCounterCompSwap atomicCounterDecrement atomicCounterExchange
atomicCounterIncrement atomicCounterMax atomicCounterMin atomicCounterOr atomicCounterSubtract atomicCounterXor
atomicExchange atomicMax atomicMin atomicOr atomicXor barrier bitCount bitfieldExtract bitfieldInsert
bitfieldReverse ceil clamp cos cosh cross dFdx dFdxCoarse dFdxFine dFdy dFdyCoarse dFdyFine degrees determinant
distance dot equal exp exp2 faceforward findLSB findMSB floatBitsToInt floatBitsToUint floor fma fract frexp
ftransform fwidth fwidthCoarse fwidthFine greaterThan greaterThanEqual groupMemoryBarrier imageAtomicAdd
imageAtomicAnd imageAtomicCompSwap imageAtomicExchange imageAtomicMax imageAtomicMin imageAtomicOr
imageAtomicXor imageLoad imageSamples imageSize imageStore imulExtended intBitsToFloat interpolateAtCentroid
interpolateAtOffset interpolateAtSample inverse inversesqrt isinf isnan ldexp length lessThan lessThanEqual log
log2 matrixCompMult max memoryBarrier memoryBarrierAtomicCounter memoryBarrierBuffer memoryBarrierImage
memoryBarrierShared min mix mod modf noise1 noise2 noise3 noise4 normalize not notEqual outerProduct
packDouble2x32 packHalf2x16 packSnorm2x16 packSnorm4x8 packUnorm2x16 packUnorm4x8 pow radians reflect refract
round roundEven shadow1D shadow1DLod shadow1DProj shadow1DProjLod shadow2D shadow2DLod shadow2DProj
shadow2DProjLod sign sin sinh smoothstep sqrt step subpassLoad tan tanh texelFetch texelFetchOffset texture
texture1D texture1DLod texture1DProj texture1DProjLod texture2D texture2DLod texture2DProj texture2DProjLod
texture3D texture3DLod texture3DProj texture3DProjLod textureCube textureCubeLod textureGather
textureGatherOffset textureGatherOffsets textureGrad textureGradOffset textureLod textureLodOffset textureOffset
textureProj textureProjGrad textureProjGradOffset textureProjLod textureProjLodOffset textureProjOffset
textureQueryLevels textureQueryLod textureSamples textureSize transpose trunc uaddCarry uintBitsToFloat
umulExtended unpackDouble2x32 unpackHalf2x16 unpackSnorm2x16 unpackSnorm4x8 unpackUnorm2x16 unpackUnorm4x8
usubBorrow
""")

# HLSL intrinsic functions and methods as DXC declares them in utils/hct/gen_intrin_main.txt
# (tests/conformance/inventory/data/hlsl.json, category builtin-function).
HLSL_FUNCTION_NAMES = _names("""
Abort AcceptHitAndEndSearch AddUint64 AllMemoryBarrier AllMemoryBarrierWithGroupSync AllocateRayQuery Append
Barrier CalculateLevelOfDetail CalculateLevelOfDetailUnclamped CallShader CandidateClusterID
CandidateGeometryIndex CandidateInstanceContributionToHitGroupIndex CandidateInstanceID CandidateInstanceIndex
CandidateObjectRayDirection CandidateObjectRayOrigin CandidateObjectToWorld3x4 CandidateObjectToWorld4x3
CandidatePrimitiveIndex CandidateProceduralPrimitiveNonOpaque CandidateTriangleBarycentrics
CandidateTriangleFrontFace CandidateTriangleObjectPosition CandidateTriangleRayT CandidateType
CandidateWorldToObject3x4 CandidateWorldToObject4x3 CheckAccessFullyMapped ClusterID CommitNonOpaqueTriangleHit
CommitProceduralPrimitiveHit CommittedClusterID CommittedGeometryIndex
CommittedInstanceContributionToHitGroupIndex CommittedInstanceID CommittedInstanceIndex
CommittedObjectRayDirection CommittedObjectRayOrigin CommittedObjectToWorld3x4 CommittedObjectToWorld4x3
CommittedPrimitiveIndex CommittedRayT CommittedStatus CommittedTriangleBarycentrics CommittedTriangleFrontFace
CommittedTriangleObjectPosition CommittedWorldToObject3x4 CommittedWorldToObject4x3 Consume Count
CreateResourceFromHeap D3DCOLORtoUBYTE4 DecrementCounter DeviceMemoryBarrier DeviceMemoryBarrierWithGroupSync
DispatchMesh DispatchRaysDimensions DispatchRaysIndex EvaluateAttributeAtSample EvaluateAttributeCentroid
EvaluateAttributeSnapped FinishedCrossGroupSharing FromRayQuery Gather GatherAlpha GatherBlue GatherCmp
GatherCmpAlpha GatherCmpBlue GatherCmpGreen GatherCmpRed GatherGreen GatherRaw GatherRed GeometryIndex
GetAttributeAtVertex GetAttributes GetDimensions GetGeometryIndex GetGroupNodeOutputRecords GetGroupWaveCount
GetGroupWaveIndex GetHitKind GetInstanceID GetInstanceIndex GetObjectRayDirection GetObjectRayOrigin
GetObjectToWorld3x4 GetObjectToWorld4x3 GetPrimitiveIndex GetRayFlags GetRayTCurrent GetRayTMin
GetRemainingRecursionLevels GetRenderTargetSampleCount GetRenderTargetSamplePosition GetSamplePosition
GetShaderTableIndex GetThreadNodeOutputRecords GetWorldRayDirection GetWorldRayOrigin GetWorldToObject3x4
GetWorldToObject4x3 GroupIncrementOutputCount GroupMemoryBarrier GroupMemoryBarrierWithGroupSync HitKind
IgnoreHit IncrementCounter InstanceID InstanceIndex InterlockedAdd InterlockedAdd64 InterlockedAnd
InterlockedAnd64 InterlockedCompareExchange InterlockedCompareExchange64 InterlockedCompareExchangeFloatBitwise
InterlockedCompareStore InterlockedCompareStore64 InterlockedCompareStoreFloatBitwise InterlockedExchange
InterlockedExchange64 InterlockedExchangeFloat InterlockedMax InterlockedMax64 InterlockedMin InterlockedMin64
InterlockedOr InterlockedOr64 InterlockedXor InterlockedXor64 Invoke IsHelperLane IsHit IsMiss IsNop IsValid
Load Load2 Load3 Load4 LoadLocalRootTableConstant MakeMiss MakeNop MaybeReorderThread NonUniformResourceIndex
ObjectRayDirection ObjectRayOrigin ObjectToWorld ObjectToWorld3x4 ObjectToWorld4x3 OutputComplete PrimitiveIndex
Proceed Process2DQuadTessFactorsAvg Process2DQuadTessFactorsMax Process2DQuadTessFactorsMin
ProcessIsolineTessFactors ProcessQuadTessFactorsAvg ProcessQuadTessFactorsMax ProcessQuadTessFactorsMin
ProcessTriTessFactorsAvg ProcessTriTessFactorsMax ProcessTriTessFactorsMin QuadAll QuadAny
QuadReadAcrossDiagonal QuadReadAcrossX QuadReadAcrossY QuadReadLaneAt RayFlags RayTCurrent RayTMin ReportHit
RestartStrip Sample SampleBias SampleCmp SampleCmpBias SampleCmpGrad SampleCmpLevel SampleCmpLevelZero
SampleGrad SampleLevel SetMeshOutputCounts SetShaderTableIndex Store Store2 Store3 Store4
ThreadIncrementOutputCount TraceRay TraceRayInline TriangleObjectPosition WaveActiveAllEqual WaveActiveAllTrue
WaveActiveAnyTrue WaveActiveBallot WaveActiveBitAnd WaveActiveBitOr WaveActiveBitXor WaveActiveCountBits
WaveActiveMax WaveActiveMin WaveActiveProduct WaveActiveSum WaveGetLaneCount WaveGetLaneIndex WaveIsFirstLane
WaveMatch WaveMultiPrefixBitAnd WaveMultiPrefixBitOr WaveMultiPrefixBitXor WaveMultiPrefixCountBits
WaveMultiPrefixProduct WaveMultiPrefixSum WavePrefixCountBits WavePrefixProduct WavePrefixSum WaveReadLaneAt
WaveReadLaneFirst WorldRayDirection WorldRayOrigin WorldToObject WorldToObject3x4 WorldToObject4x3
WriteSamplerFeedback WriteSamplerFeedbackBias WriteSamplerFeedbackGrad WriteSamplerFeedbackLevel abort abs acos
all and any asdouble asfloat asfloat16 asin asint asint16 asuint asuint16 atan atan2 ceil clamp clip cos cosh
countbits cross ddx ddx_coarse ddx_fine ddy ddy_coarse ddy_fine degrees determinant distance dot dot2add
dot4add_i8packed dot4add_u8packed dst errorf exp exp2 f16tof32 f32tof16 faceforward firstbithigh firstbitlow
floor fma fmod frac frexp fwidth isfinite isinf isnan isnormal ldexp length lerp lit log log10 log2 mad max min
modf msad4 mul noise normalize or pack_clamp_s8 pack_clamp_u8 pack_s8 pack_u8 pow printf radians rcp reflect
refract reversebits round rsqrt saturate select sign sin sincos sinh smoothstep source_mark sqrt step tan tanh
tex1D tex1Dbias tex1Dgrad tex1Dlod tex1Dproj tex2D tex2Dbias tex2Dgrad tex2Dlod tex2Dproj tex3D tex3Dbias
tex3Dgrad tex3Dlod tex3Dproj texCUBE texCUBEbias texCUBEgrad texCUBElod texCUBEproj transpose trunc unpack_s8s16
unpack_s8s32 unpack_u8u16 unpack_u8u32
""")

# Metal Shading Language 4.1 chapter 6 standard library functions, unqualified
# (tests/conformance/inventory/data/metal.json, category builtin-function).
METAL_FUNCTION_NAMES = _names("""
abort abs absdiff accept_any_intersection acos acosh addsat all any as_const as_storage_type asin asinh
assume_curve_basis assume_curve_control_point_count assume_curve_type assume_geometry_type
assume_identity_transforms atan atan2 atanh atomic_compare_exchange_weak atomic_compare_exchange_weak_explicit
atomic_exchange atomic_exchange_explicit atomic_fetch_add atomic_fetch_add_explicit atomic_fetch_and
atomic_fetch_and_explicit atomic_fetch_max atomic_fetch_max_explicit atomic_fetch_min atomic_fetch_min_explicit
atomic_fetch_or atomic_fetch_or_explicit atomic_fetch_sub atomic_fetch_sub_explicit atomic_fetch_xor
atomic_fetch_xor_explicit atomic_load atomic_load_explicit atomic_max atomic_max_explicit atomic_min
atomic_min_explicit atomic_store atomic_store_explicit atomic_thread_fence back begin bezier bezier_derivative
bezier_second_derivative block_read bspline bspline_derivative bspline_second_derivative calculate_clamped_lod
calculate_unclamped_lod category catmull_rom catmull_rom_derivative catmull_rom_second_derivative ceil clamp
clear_barrier clz commit_bounding_box_intersection commit_curve_intersection commit_triangle_intersection
concurrent_dispatch_threadgroups concurrent_dispatch_threads copy_command copysign cos cosh cospi cross ctz data
declval deinterleave denorm_min determinant dfdx dfdy discard_fragment distance distance_squared divide dot
draw_indexed_patches draw_indexed_primitives draw_mesh_threadgroups draw_mesh_threads draw_patches
draw_primitives empty end epsilon exp exp10 exp2 extent extract extract_bits fabs faceforward fdim fence floor
fma fmax fmax3 fmedian3 fmin fmin3 fmod force_opacity fract frexp front fwidth gather gather_compare get
get_array_size get_block_size get_buffer get_candidate_curve_distance get_candidate_curve_parameter
get_candidate_geometry_id get_candidate_instance_count get_candidate_instance_id get_candidate_intersection_type
get_candidate_object_to_world_transform get_candidate_primitive_data get_candidate_primitive_id
get_candidate_ray_direction get_candidate_ray_origin get_candidate_triangle_barycentric_coord
get_candidate_triangle_distance get_candidate_user_instance_id get_candidate_world_to_object_transform
get_capacity get_color_coverage_mask get_committed_curve_parameter get_committed_distance
get_committed_geometry_id get_committed_instance_count get_committed_instance_id get_committed_intersection_type
get_committed_object_to_world_transform get_committed_primitive_data get_committed_primitive_id
get_committed_ray_direction get_committed_ray_origin get_committed_triangle_barycentric_coord
get_committed_user_instance_id get_committed_world_to_object_transform get_curve_parameter get_depth
get_distance get_extent get_function_id get_geometry_id get_height get_instance_count get_instance_id
get_intersection_params get_iterator get_multidimensional_index get_num_colors get_num_mip_levels
get_num_samples get_object_to_world_transform get_primitive_data get_primitive_id get_rank get_ray_direction
get_ray_min_distance get_ray_origin get_sample_position get_static_extent get_stride get_stride_bytes
get_triangle_barycentric_coord get_type get_user_instance_id get_visible_function_table get_width
get_world_space_ray_direction get_world_space_ray_origin get_world_to_object_transform hadd has_static_extent
hermite hermite_derivative hermite_second_derivative ilogb infinity insert_bits interleave interpolate_at_center
interpolate_at_centroid interpolate_at_offset interpolate_at_sample intersect
is_candidate_non_opaque_bounding_box is_candidate_triangle_front_facing is_committed_triangle_front_facing
is_null_acceleration_structure is_null_function_handle is_null_instance_acceleration_structure
is_null_intersection_function_table is_null_primitive_acceleration_structure is_null_tensor is_null_texture
is_null_visible_function_table is_triangle_front_facing is_valid_element isfinite isinf isnan isnormal isordered
isunordered ldexp length length_squared load log log10 log2 log_debug log_error log_fault log_info log_with_type
lowest mad24 madhi madsat make_array_ref make_filled_matrix make_filled_simdgroup_matrix make_uniform
map_iterator map_physical_to_screen_coordinates map_screen_to_physical_coordinates max max3 max_size median3 min
min3 mix modf move mul24 mulhi next nextafter normalize pack pack_float_to_snorm10a2 pack_float_to_snorm2x16
pack_float_to_snorm4x8 pack_float_to_srgb_unorm4x8 pack_float_to_unorm10a2 pack_float_to_unorm2x16
pack_float_to_unorm4x8 pack_float_to_unorm565 pack_half_to_snorm10a2 pack_half_to_snorm2x16
pack_half_to_snorm4x8 pack_half_to_srgb_unorm4x8 pack_half_to_unorm10a2 pack_half_to_unorm2x16
pack_half_to_unorm4x8 pack_half_to_unorm565 popcount pow powr quad_active_threads_mask quad_all quad_and
quad_any quad_ballot quad_broadcast quad_broadcast_first quad_is_first quad_is_helper_thread quad_max quad_min
quad_or quad_prefix_exclusive_product quad_prefix_exclusive_sum quad_prefix_inclusive_product
quad_prefix_inclusive_sum quad_product quad_shuffle quad_shuffle_and_fill_down quad_shuffle_and_fill_up
quad_shuffle_down quad_shuffle_rotate_down quad_shuffle_rotate_up quad_shuffle_up quad_shuffle_xor quad_sum
quad_xor quiet_NaN rank rank_dynamic read reflect refract reset resident reverse_bits rhadd rint rotate round
round_error rsqrt sample sample_compare saturate select set set_barrier set_base_id set_buffer
set_compute_pipeline_state set_cull_mode set_depth_bias set_depth_clip_mode set_depth_stencil_state
set_fragment_buffer set_front_facing_winding set_geometry_cull_mode set_geometry_multiplier set_imageblock_size
set_index set_indices set_kernel_buffer set_mesh_buffer set_object_buffer set_object_threadgroup_memory_length
set_opacity_cull_mode set_primitive set_primitive_count set_render_pipeline_state set_stage_in_region
set_threadgroup_memory_length set_threadgroups_per_grid set_triangle_cull_mode set_triangle_fill_mode
set_triangle_front_facing_winding set_vertex set_vertex_buffer set_visible_function_table sign signaling_NaN
signbit simd_active_threads_mask simd_all simd_and simd_any simd_ballot simd_broadcast simd_broadcast_first
simd_is_first simd_is_helper_thread simd_max simd_min simd_or simd_prefix_exclusive_product
simd_prefix_exclusive_sum simd_prefix_inclusive_product simd_prefix_inclusive_sum simd_product simd_shuffle
simd_shuffle_and_fill_down simd_shuffle_and_fill_up simd_shuffle_down simd_shuffle_rotate_down
simd_shuffle_rotate_up simd_shuffle_up simd_shuffle_xor simd_sum simd_xor simdgroup_barrier simdgroup_load
simdgroup_multiply simdgroup_multiply_accumulate simdgroup_store sin sincos sinh sinpi size slice smoothstep
sparse_block_read sparse_gather sparse_gather_compare sparse_read sparse_sample sparse_sample_compare sqrt
static_extent step store subsat subsystem tan tanh tanpi thread_elements threadgroup_barrier transpose trunc
unpack unpack_snorm10a2_to_float unpack_snorm10a2_to_half unpack_snorm2x16_to_float unpack_snorm2x16_to_half
unpack_snorm4x8_to_float unpack_snorm4x8_to_half unpack_unorm10a2_to_float unpack_unorm10a2_to_half
unpack_unorm2x16_to_float unpack_unorm2x16_to_half unpack_unorm4x8_srgb_to_float unpack_unorm4x8_srgb_to_half
unpack_unorm4x8_to_float unpack_unorm4x8_to_half unpack_unorm565_to_float unpack_unorm565_to_half value write
""")

# Functions of the GLSL extensions the GLSL and SPIR-V importers read: GL_KHR_shader_subgroup (basic barriers),
# GL_EXT_nonuniform_qualifier, GL_EXT_debug_printf, GL_EXT_ray_tracing, GL_EXT_ray_query,
# GL_EXT_ray_tracing_position_fetch, GL_EXT_mesh_shader, GL_ARB_sparse_texture2, GL_ARB_sparse_texture_clamp,
# GL_ARB_fragment_shader_interlock, GL_ARB_shader_clock and GL_EXT_shader_realtime_clock.
GLSL_EXTENSION_FUNCTION_NAMES = _names("""
subgroupBarrier subgroupMemoryBarrier subgroupMemoryBarrierBuffer subgroupMemoryBarrierShared
subgroupMemoryBarrierImage subgroupElect nonuniformEXT debugPrintfEXT traceRayEXT reportIntersectionEXT
executeCallableEXT ignoreIntersectionEXT terminateRayEXT rayQueryInitializeEXT rayQueryProceedEXT
rayQueryTerminateEXT rayQueryGenerateIntersectionEXT rayQueryConfirmIntersectionEXT rayQueryGetRayFlagsEXT
rayQueryGetRayTMinEXT rayQueryGetWorldRayOriginEXT rayQueryGetWorldRayDirectionEXT rayQueryGetIntersectionTypeEXT
rayQueryGetIntersectionTEXT rayQueryGetIntersectionInstanceCustomIndexEXT rayQueryGetIntersectionInstanceIdEXT
rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT rayQueryGetIntersectionGeometryIndexEXT
rayQueryGetIntersectionPrimitiveIndexEXT rayQueryGetIntersectionBarycentricsEXT rayQueryGetIntersectionFrontFaceEXT
rayQueryGetIntersectionCandidateAABBOpaqueEXT rayQueryGetIntersectionObjectRayDirectionEXT
rayQueryGetIntersectionObjectRayOriginEXT rayQueryGetIntersectionObjectToWorldEXT
rayQueryGetIntersectionWorldToObjectEXT rayQueryGetIntersectionTriangleVertexPositionsEXT SetMeshOutputsEXT
EmitMeshTasksEXT sparseTextureARB sparseTextureLodARB sparseTextureOffsetARB sparseTexelFetchARB
sparseTexelFetchOffsetARB sparseTextureLodOffsetARB sparseTextureGradARB sparseTextureGradOffsetARB
sparseTextureGatherARB sparseTextureGatherOffsetARB sparseTextureGatherOffsetsARB sparseImageLoadARB
sparseTexelsResidentARB sparseTextureClampARB sparseTextureOffsetClampARB sparseTextureGradClampARB
sparseTextureGradOffsetClampARB textureClampARB textureOffsetClampARB textureGradClampARB textureGradOffsetClampARB
beginInvocationInterlockARB endInvocationInterlockARB clockARB clock2x32ARB clockRealtimeEXT clockRealtime2x32EXT
""")

# MSL spellings outside the library function tables: ``as_type<T>(x)`` (MSL 4.1 section 2.23, an operator in the
# MSL inventory), the ``max_anisotropy`` and ``lod_clamp`` arguments of a sampler constructor (section 2.10), and the
# pi-scaled inverse trigonometric functions of ``metal_math`` that :mod:`.metal_math_spellings` reduces.
_METAL_OTHER_NAMES = (
    _names("as_type")
    | frozenset(SAMPLER_STATE_CALL_ARGUMENTS)
    | frozenset(PI_FUNCTIONS)
)

# Identifiers that begin with two underscores are reserved to the implementation (C++14 [lex.name] paragraph 3, which
# MSL and CUDA inherit; GLSL 4.60 section 3.7), so a program never declares one. They name Clang builtins
# (``__builtin_ctz``), CUDA intrinsics (``__sincosf``) and CrossGL's rule helpers (``__cgl_wrap_add``), which the
# importers and :mod:`crosstl.translator.semantics` produce and the generators lower or refuse.
RESERVED_NAME_PREFIX = "__"

# CUDA's and HIP's device library functions (:mod:`~crosstl.translator.codegen.cuda_library_functions`), HIP's
# ``atomicAddNoRet``, and CUDA's warp shuffle, vote, match, reduce and synchronization functions (CUDA C++
# Programming Guide, "Warp Shuffle Functions", "Warp Vote Functions", "Warp Match Functions", "Warp Reduce
# Functions"), which :mod:`.warp` emulates or refuses for each target.
CUDA_FUNCTION_NAMES = (
    frozenset(CUDA_LIBRARY_FUNCTIONS)
    | _names("atomicAddNoRet")
    | frozenset(SHUFFLES)
    | frozenset(OTHER_WARP_INTRINSICS)
)

# CrossGL operations the importers produce and the generators lower: atomics on CrossGL atomic values and fences, the
# barrier family, sparse residency and sampler feedback writes, fine and coarse derivative widths, bit
# reinterpretation of 64-bit values, debug output, fragment discard, ``inverseSqrt``, the ``sizeof`` and ``alignof``
# operators (with ``metal_sizeof``, a size in the MSL layout, :mod:`.sizeof_values`), lambdas, Mojo's ``assert`` and
# ``raise``, and the constructor and helpers of the built-in ``Option<T>`` that the generators declare when a
# program uses them.
_CROSSGL_OPERATION_NAMES = _names("""
atomicLoad atomicStore atomicCompareExchange atomicCompareExchangeWeak atomicThreadFence workgroupBarrier
workgroupExecutionBarrier subgroupExecutionBarrier deviceMemoryBarrier allMemoryBarrier textureResidency
textureResident textureGatherCompareOffsets write_sampler_feedback write_sampler_feedback_bias
write_sampler_feedback_grad write_sampler_feedback_level fwidth_fine fwidth_coarse doubleBitsToLong longBitsToDouble
printf discard inverseSqrt alignof lambda assert raise Some is_Some is_None unwrap_Some
""") | frozenset(SIZEOF_LAYOUTS)

# CrossGL's intrinsic functions: the wave, subgroup, ray tracing and mesh intrinsics the parser and
# :mod:`~crosstl.translator.wave_operations` define, the buffer, image and texture resource intrinsics, and the
# operations above.
CROSSGL_FUNCTION_NAMES = (
    frozenset(_parser.WAVE_INTRINSICS)
    | frozenset(_parser.RAYTRACING_INTRINSICS)
    | frozenset(_parser.MESH_INTRINSICS)
    | frozenset(_wave_operations.GLSL_SUBGROUP_FUNCTIONS)
    | frozenset(_wave_operations.WAVE_EXTENDED_OPERATIONS)
    | frozenset(_validation.IMAGE_RESOURCE_INTRINSIC_NAMES)
    | frozenset(RESERVED_GENERIC_FUNCTION_BUILTIN_NAMES)
    | _CROSSGL_OPERATION_NAMES
)

# Type templates a call constructs or converts to with explicit arguments: CrossGL's and MSL's ``vec<T, N>`` and
# ``matrix<T, C, R>`` (MSL 4.1 sections 2.2 and 2.3), HLSL's ``vector<T, N>`` and ``matrix<T, R, C>``, and the alias
# templates of MSL's ``metal_type_traits`` header (``static_cast<remove_cv_t<U>>(x)``).
TYPE_TEMPLATE_NAMES = _names("""
vec vector matrix add_const_t add_pointer_t common_type_t conditional_t enable_if_t make_scalar_t make_signed_t
make_unsigned_t remove_addrspace_t remove_coherent_t remove_const_t remove_cv_t remove_extent_t remove_pointer_t
remove_reference_t remove_volatile_t underlying_type_t
""")

BUILTIN_FUNCTION_NAMES = (
    GLSL_FUNCTION_NAMES
    | GLSL_EXTENSION_FUNCTION_NAMES
    | HLSL_FUNCTION_NAMES
    | METAL_FUNCTION_NAMES
    | _METAL_OTHER_NAMES
    | CUDA_FUNCTION_NAMES
    | CROSSGL_FUNCTION_NAMES
)
