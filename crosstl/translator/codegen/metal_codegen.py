"""CrossGL-to-Metal code generator."""

import ast as py_ast
import re
from hashlib import sha1

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ...backend.common_ast import ReturnNode as BackendReturnNode
from ...backend.common_ast import VariableNode as BackendVariableNode
from ...glsl_builtins import GLSL_BUILTIN_INT_LIMITS
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ConstructorNode,
    ContinueNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerAccessNode,
    PointerType,
    PreprocessorNode,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    ReturnNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from ..validation import (
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    INTEGER_COORDINATE_INTRINSIC_NAMES,
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_member_names,
    collect_duplicate_cbuffer_names,
    collect_non_resource_global_resource_shadows,
    expression_debug_name,
    texture_bias_argument_index,
    texture_compare_argument_index,
    texture_gather_component_argument_index,
    texture_gradient_argument_indices,
    texture_lod_argument_index,
    texture_mip_level_argument_index,
    texture_offset_argument_indices,
    texture_query_lod_coordinate_argument_index,
    texture_sample_index_argument_index,
)
from .array_utils import (
    collect_literal_int_constants,
    collect_struct_member_types,
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    get_array_size_from_node,
    parse_array_type,
    split_array_type_suffix,
)
from .constant_ordering import partition_constants_by_struct_dependency
from .enum_utils import (
    build_generic_enum_specialization,
    collect_enum_struct_variant_fields,
    collect_enum_type_names,
    collect_enum_variant_constants,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_struct_fields,
    enum_value_expression,
    enum_variant_constructor_name,
    enum_variant_payload_fields,
    generate_enum_constants,
    generate_enum_constructor_call,
    generate_enum_constructor_expression,
    generate_generic_enum_constants,
    generic_enum_specialized_fields,
    generic_enum_specialized_type_name,
    generic_enum_specialized_variant_fields,
    generic_type_parts,
    infer_enum_constructor_type,
)
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
)
from .generic_struct_utils import (
    build_generic_struct_specialization,
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    format_struct_constructor_expression,
    generate_struct_constructor_expression,
    generic_struct_specialized_fields,
    generic_struct_specialized_type_name,
    infer_struct_constructor_type,
    normalize_specialization_type_text,
)
from .glsl_buffer_layout import (
    byte_offset_add,
    byte_offset_expression,
    collect_lowered_glsl_buffer_blocks,
    glsl_buffer_block_node_type,
    glsl_buffer_compound_binary_operator,
    matrix_column_offsets,
    vector_component_offsets,
)
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    default_storage_image_channel_count,
    explicit_image_access,
    explicit_image_format,
    floating_coordinate_dimension_from_type_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_atomic_helper_descriptor_fields,
    image_atomic_helper_resource_metadata,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
)
from .image_access_contracts import (
    image_atomic_value_arguments as shared_image_atomic_value_arguments,
)
from .image_access_contracts import (
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    image_format_channel_count,
    image_format_component_kind,
    image_format_component_type,
    image_format_or_default_channel_count,
    image_format_result_type,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_load_result_shape_mismatch,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_resource_metadata,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    image_store_value_shape_mismatch,
    integer_coordinate_dimension_from_type_name,
    is_floating_scalar_type_name,
    is_image_atomic_operation,
    is_image_format_attribute,
    is_image_resource_operation,
    is_integer_coordinate_type_name,
    is_integer_scalar_type_name,
    is_metal_float_image_resource,
    is_metal_integer_image_type,
    is_metal_storage_image_resource,
    is_numeric_scalar_type_name,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_basic_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_grad_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_lod_operation,
    is_projected_texture_operation,
    is_resource_access_attribute,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_scalar_image_format,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_basic_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_operation,
    is_texture_gather_single_offset_operation,
    is_texture_query_levels_operation,
    is_texture_query_lod_operation,
    is_texture_sample_basic_offset_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_offset_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_lod_offset_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_sample_operation,
    is_texture_sampling_operation,
    is_two_component_image_format,
    metal_storage_image_access_agnostic_type,
    metal_storage_image_component_type,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    projected_texture_extra_argument_count_error,
    projected_texture_offset_capability_error,
    record_explicit_image_metadata,
    requires_integer_coordinate,
    resolve_image_atomic_component_kind,
    resource_query_method_size_descriptor,
    should_validate_image_load_result_shape,
    storage_image_atomic_zero_value,
    storage_image_format_store_constructor,
    storage_image_load_component_suffix,
    storage_image_store_constructors,
    storage_image_store_value_expression,
    storage_image_two_component_store_expression,
    storage_image_zero_values,
    supported_image_formats,
)
from .image_access_contracts import (
    texture_argument_diagnostic_type as shared_texture_argument_diagnostic_type,
)
from .image_access_contracts import (
    texture_compare_argument_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
    texture_gather_capability_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    texture_image_resource_operation_names,
    texture_multisample_sample_type_error,
    texture_query_levels_multisample_expression,
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_swizzle,
    texture_query_lod_coordinate_type_error,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
    texture_sample_offset_capability_error,
    texture_sample_offset_extra_argument_count_error,
    texture_samples_query_expression,
    unsupported_cube_texel_fetch_expression,
    unsupported_image_atomic_expression,
    unsupported_multisample_image_atomic_expression,
    unsupported_multisample_image_store_expression,
    unsupported_multisample_texel_fetch_offset_expression,
    unsupported_multisample_texture_call_vector_expression,
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_projected_texture_call_expression,
    unsupported_projected_texture_operation_error,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texture_compare_operation_error,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_gather_compare_call_expression,
    unsupported_texture_offset_call_expression,
    unsupported_texture_offset_operation_error,
    unsupported_texture_query_levels_expression,
    unsupported_texture_query_lod_expression,
    unsupported_texture_samples_query_call_expression,
    validate_texture_operation_arity,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)
from .resource_arrays import collect_resource_array_size_hints
from .stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    collect_stage_local_structs,
    collect_stage_local_variables,
    compute_local_size,
    deduplicate_named_declarations,
    function_stage_name,
    normalize_stage_name,
    order_functions_by_dependencies,
    should_emit_qualified_function,
    stage_layout_entry_value,
    stage_matches,
)

FRAGMENT_INVOCATION_DENSITY_CAPABILITIES = (
    "glsl.extension.GL_EXT_fragment_invocation_density",
    "glsl.builtin.gl_FragSizeEXT",
)


class UnsupportedMetalFeatureError(ValueError):
    """Raised when Metal has no equivalent target representation."""

    project_diagnostic_code = "project.translate.unsupported-feature"

    def __init__(
        self,
        feature,
        message,
        *,
        missing_capabilities=(),
        operation=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.feature = feature
        self.missing_capabilities = tuple(missing_capabilities)
        self.operation = operation
        self.reason = reason
        self.source_location = source_location


class CharTypeMapper:
    """Normalize CrossGL char-like scalar and vector types for Metal output."""

    def map_char_type(self, vtype):
        """Return the Metal-compatible integer type for a char-like type."""
        char_type_mapping = {
            "char": "int",
            "signed char": "int",
            "unsigned char": "uint",
            "char2": "int2",
            "char3": "int3",
            "char4": "int4",
            "uchar2": "uint2",
            "uchar3": "uint3",
            "uchar4": "uint4",
        }
        return char_type_mapping.get(vtype, vtype)


class MetalCodeGen:
    """Emit Metal Shading Language from the shared CrossGL translator AST."""

    BINARY_PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        ">": 7,
        "<=": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    ASSOCIATIVE_BINARY_OPS = {"+", "*", "&&", "||", "&", "|", "^"}
    METAL_ATOMIC_FENCE_MEMORY_FLAGS = frozenset(
        {
            "mem_none",
            "mem_device",
            "mem_threadgroup",
            "mem_texture",
            "mem_threadgroup_imageblock",
            "mem_object_data",
        }
    )
    METAL_ATOMIC_FENCE_MEMORY_ORDERS = frozenset(
        {
            "memory_order_relaxed",
            "memory_order_acquire",
            "memory_order_release",
            "memory_order_acq_rel",
            "memory_order_seq_cst",
        }
    )
    METAL_ATOMIC_FENCE_THREAD_SCOPES = frozenset(
        {
            "thread_scope_thread",
            "thread_scope_simdgroup",
            "thread_scope_threadgroup",
            "thread_scope_device",
            "thread_scope_system",
        }
    )
    METAL_COOPERATIVE_MATRIX_FUNCTIONS = {
        "load": "simdgroup_load",
        "store": "simdgroup_store",
        "multiply_accumulate": "simdgroup_multiply_accumulate",
    }
    METAL_RESERVED_LOCAL_IDENTIFIERS = {
        "kernel",
        "vertex",
        "fragment",
        "compute",
        "intersection",
        "anyhit",
        "closesthit",
        "miss",
        "callable",
        "mesh",
        "object",
        "amplification",
    }
    METAL_INTERPOLATION_ATTRIBUTES = {
        "center_no_perspective": "center_no_perspective",
        "center_perspective": "center_perspective",
        "centroid": "centroid_perspective",
        "centroid_no_perspective": "centroid_no_perspective",
        "centroid_perspective": "centroid_perspective",
        "flat": "flat",
        "linear_centroid": "centroid_perspective",
        "linear_noperspective": "center_no_perspective",
        "linear_noperspective_centroid": "centroid_no_perspective",
        "linear_sample": "sample_perspective",
        "nointerpolation": "flat",
        "noperspective": "center_no_perspective",
        "sample": "sample_perspective",
        "sample_no_perspective": "sample_no_perspective",
        "sample_perspective": "sample_perspective",
    }
    METAL_DERIVATIVE_FUNCTION_ALIASES = {
        "ddx": "dfdx",
        "dFdx": "dfdx",
        "ddy": "dfdy",
        "dFdy": "dfdy",
    }
    METAL_STDLIB_BUILTIN_FUNCTIONS = {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "clamp",
        "cos",
        "cross",
        "distance",
        "dot",
        "exp",
        "exp2",
        "faceforward",
        "floor",
        "fract",
        "length",
        "log",
        "log2",
        "max",
        "min",
        "mix",
        "normalize",
        "pow",
        "reflect",
        "refract",
        "round",
        "rsqrt",
        "saturate",
        "sign",
        "sin",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "trunc",
    }
    METAL_BITCAST_FUNCTION_TARGETS = {
        "asfloat": "float",
        "asint": "int",
        "asuint": "uint",
        "floatBitsToInt": "int",
        "floatBitsToUint": "uint",
        "intBitsToFloat": "float",
        "uintBitsToFloat": "float",
    }
    METAL_INTEGER_BIT_FUNCTION_TARGETS = {
        "bitCount": "popcount",
        "bitfieldReverse": "reverse_bits",
        "countbits": "popcount",
        "reversebits": "reverse_bits",
    }
    METAL_WAVE_INTRINSIC_ARITIES = {
        "WaveGetLaneCount": 0,
        "WaveGetLaneIndex": 0,
        "WaveIsFirstLane": 0,
        "WaveActiveSum": 1,
        "WaveActiveProduct": 1,
        "WaveActiveBitAnd": 1,
        "WaveActiveBitOr": 1,
        "WaveActiveBitXor": 1,
        "WaveActiveMin": 1,
        "WaveActiveMax": 1,
        "WaveActiveAllTrue": 1,
        "WaveActiveAnyTrue": 1,
        "WaveActiveAllEqual": 1,
        "WaveActiveCountBits": 1,
        "WaveActiveBallot": 1,
        "WaveReadLaneAt": 2,
        "WaveReadLaneFirst": 1,
        "WaveShuffleDown": 2,
        "WaveShuffleUp": 2,
        "WaveShuffleAndFillUp": 3,
        "WaveShuffleXor": 2,
        "WavePrefixSum": 1,
        "WavePrefixProduct": 1,
        "WavePrefixInclusiveSum": 1,
        "WavePrefixInclusiveProduct": 1,
        "WavePrefixCountBits": 1,
        "WaveMatch": 1,
        "WaveMultiPrefixSum": 2,
        "WaveMultiPrefixCountBits": 2,
        "WaveMultiPrefixProduct": 2,
        "WaveMultiPrefixBitAnd": 2,
        "WaveMultiPrefixBitOr": 2,
        "WaveMultiPrefixBitXor": 2,
        "QuadReadAcrossX": 1,
        "QuadReadAcrossY": 1,
        "QuadReadAcrossDiagonal": 1,
        "QuadReadLaneAt": 2,
        "QuadAny": 1,
        "QuadAll": 1,
    }
    METAL_WAVE_DIRECT_MAPPINGS = {
        "WaveActiveSum": "simd_sum",
        "WaveActiveProduct": "simd_product",
        "WaveActiveBitAnd": "simd_and",
        "WaveActiveBitOr": "simd_or",
        "WaveActiveBitXor": "simd_xor",
        "WaveActiveMin": "simd_min",
        "WaveActiveMax": "simd_max",
        "WaveActiveAllTrue": "simd_all",
        "WaveActiveAnyTrue": "simd_any",
        "WaveReadLaneFirst": "simd_broadcast_first",
        "WaveShuffleDown": "simd_shuffle_down",
        "WaveShuffleUp": "simd_shuffle_up",
        "WaveShuffleAndFillUp": "simd_shuffle_and_fill_up",
        "WaveShuffleXor": "simd_shuffle_xor",
        "WavePrefixSum": "simd_prefix_exclusive_sum",
        "WavePrefixProduct": "simd_prefix_exclusive_product",
        "WavePrefixInclusiveSum": "simd_prefix_inclusive_sum",
        "WavePrefixInclusiveProduct": "simd_prefix_inclusive_product",
        "QuadAny": "quad_any",
        "QuadAll": "quad_all",
    }
    METAL_WAVE_UNSUPPORTED_OPERATIONS = {}
    METAL_WAVE_MULTI_PREFIX_INTRINSICS = {
        "WaveMultiPrefixSum",
        "WaveMultiPrefixCountBits",
        "WaveMultiPrefixProduct",
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    METAL_WAVE_MULTI_PREFIX_NUMERIC_INTRINSICS = {
        "WaveMultiPrefixSum",
        "WaveMultiPrefixProduct",
    }
    METAL_WAVE_MULTI_PREFIX_INTEGER_INTRINSICS = {
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    METAL_WAVE_MULTI_PREFIX_HELPERS = {
        "WaveMultiPrefixSum": "__crossgl_metal_wave_multi_prefix_sum",
        "WaveMultiPrefixCountBits": "__crossgl_metal_wave_multi_prefix_count_bits",
        "WaveMultiPrefixProduct": "__crossgl_metal_wave_multi_prefix_product",
        "WaveMultiPrefixBitAnd": "__crossgl_metal_wave_multi_prefix_bit_and",
        "WaveMultiPrefixBitOr": "__crossgl_metal_wave_multi_prefix_bit_or",
        "WaveMultiPrefixBitXor": "__crossgl_metal_wave_multi_prefix_bit_xor",
    }
    METAL_WAVE_BOOL_ARGUMENT_INTRINSICS = {
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "QuadAny",
        "QuadAll",
    }
    METAL_WAVE_NUMERIC_VALUE_INTRINSICS = {
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMin",
        "WaveActiveMax",
        "WavePrefixSum",
        "WavePrefixProduct",
        "WavePrefixInclusiveSum",
        "WavePrefixInclusiveProduct",
    }
    METAL_WAVE_INTEGER_VALUE_INTRINSICS = {
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
    }
    METAL_WAVE_SIMDGROUP_VALUE_INTRINSICS = {
        "WaveReadLaneAt",
        "WaveReadLaneFirst",
        "WaveActiveAllEqual",
        "QuadReadAcrossX",
        "QuadReadAcrossY",
        "QuadReadAcrossDiagonal",
        "QuadReadLaneAt",
    }
    METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS = {"WaveShuffleAndFillUp"}
    METAL_WAVE_UINT_RESULT_INTRINSICS = {
        "WaveGetLaneCount",
        "WaveGetLaneIndex",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "WaveMultiPrefixCountBits",
    }
    METAL_WAVE_BOOL_RESULT_INTRINSICS = {
        "WaveIsFirstLane",
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveAllEqual",
        "QuadAny",
        "QuadAll",
    }
    METAL_WAVE_UINT4_RESULT_INTRINSICS = {
        "WaveActiveBallot",
        "WaveMatch",
    }
    METAL_WAVE_VALUE_RESULT_INTRINSICS = (
        METAL_WAVE_NUMERIC_VALUE_INTRINSICS
        | METAL_WAVE_INTEGER_VALUE_INTRINSICS
        | METAL_WAVE_SIMDGROUP_VALUE_INTRINSICS
        | METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS
        | {
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }
    )
    METAL_WAVE_NUMERIC_COMPONENT_TYPES = {"float", "half", "int", "uint"}
    METAL_WAVE_INTEGER_COMPONENT_TYPES = {"int", "uint"}
    METAL_GLSL_SUBGROUP_LANE_BUILTINS = {
        "gl_SubgroupInvocationID": (
            "thread_index_in_simdgroup",
            "uint",
        ),
        "gl_SubgroupSize": (
            "threads_per_simdgroup",
            "uint",
        ),
    }
    METAL_RAY_FLAG_VALUES = {
        "RAY_FLAG_NONE": 0x00,
        "RAY_FLAG_FORCE_OPAQUE": 0x01,
        "RAY_FLAG_FORCE_NON_OPAQUE": 0x02,
        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH": 0x04,
        "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER": 0x08,
        "RAY_FLAG_CULL_BACK_FACING_TRIANGLES": 0x10,
        "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES": 0x20,
        "RAY_FLAG_CULL_OPAQUE": 0x40,
        "RAY_FLAG_CULL_NON_OPAQUE": 0x80,
        "RAY_FLAG_SKIP_TRIANGLES": 0x100,
        "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES": 0x200,
    }
    METAL_RAY_QUERY_RETURN_TYPES = {
        "Proceed": "bool",
        "Abort": "void",
        "Terminate": "void",
        "ConfirmIntersection": "void",
        "CommitNonOpaqueTriangleHit": "void",
        "GenerateIntersection": "void",
        "CommitProceduralPrimitiveHit": "void",
        "TraceRayInline": "void",
        "CandidateType": "uint",
        "CommittedType": "uint",
        "CommittedStatus": "uint",
        "CandidatePrimitiveIndex": "uint",
        "CommittedPrimitiveIndex": "uint",
        "CandidateInstanceID": "uint",
        "CommittedInstanceID": "uint",
        "CandidateInstanceIndex": "uint",
        "CommittedInstanceIndex": "uint",
        "CandidateGeometryIndex": "uint",
        "CommittedGeometryIndex": "uint",
        "CandidateInstanceCustomIndex": "uint",
        "CommittedInstanceCustomIndex": "uint",
        "CandidateInstanceShaderBindingTableRecordOffset": "uint",
        "CommittedInstanceShaderBindingTableRecordOffset": "uint",
        "RayFlags": "uint",
        "CandidateObjectRayOrigin": "float3",
        "CandidateObjectRayDirection": "float3",
        "CommittedObjectRayOrigin": "float3",
        "CommittedObjectRayDirection": "float3",
        "WorldRayOrigin": "float3",
        "WorldRayDirection": "float3",
        "CandidateRayT": "float",
        "CommittedRayT": "float",
        "CandidateObjectRayTMin": "float",
        "RayTMin": "float",
        "CandidateTriangleBarycentrics": "float2",
        "CommittedTriangleBarycentrics": "float2",
        "CandidateTriangleFrontFace": "bool",
        "CommittedTriangleFrontFace": "bool",
        "CandidateAABBOpaque": "bool",
        "CandidateObjectToWorld3x4": "float3x4",
        "CandidateWorldToObject3x4": "float3x4",
        "CommittedObjectToWorld3x4": "float3x4",
        "CommittedWorldToObject3x4": "float3x4",
        "CandidateTriangleVertexPositions": "void",
        "CommittedTriangleVertexPositions": "void",
    }

    def __init__(self):
        """Initialize Metal type maps and per-generation resource state."""
        self.current_shader = None
        self.vertex_item = None
        self.fragment_item = None
        self.gl_position = False
        self.char_mapper = CharTypeMapper()
        self.texture_variables = []
        self.acceleration_structure_variables = []
        self.visible_function_table_variables = []
        self.intersection_function_table_variables = []
        self.unsupported_metal_acceleration_structure_array_variables = {}
        self.unsupported_metal_ray_function_table_array_variables = {}
        self.sampler_variables = []
        self.structured_buffer_variables = []
        self.structured_buffer_length_variables = []
        self.structured_buffer_counter_variables = []
        self.metal_buffer_resource_variables = []
        self.metal_resource_binding_indices_by_id = {}
        self.metal_source_binding_stage_by_id = {}
        self.metal_program_scope_value_globals = set()
        self.metal_program_scope_groupshared_globals = set()
        self.metal_lowered_program_scope_groupshared_globals_by_function = {}
        self.metal_lowered_program_scope_groupshared_global_ids = set()
        self.metal_program_scope_value_global_types = {}
        self.cbuffer_variables = []
        self.cbuffer_binding_indices = {}
        self.cbuffer_parameter_names = {}
        self.cbuffer_member_references = {}
        self.ambiguous_cbuffer_members = set()
        self.hlsl_program_constant_global_ids = set()
        self.cbuffers_by_name = {}
        self.user_function_names = set()
        self.function_parameter_names = {}
        self.function_parameter_infos = {}
        self.function_parameter_nodes = {}
        self.functions_by_name = {}
        self.metal_skipped_function_parameter_indices = {}
        self.function_return_types = {}
        self.function_image_access_requirements = {}
        self.function_cbuffer_dependencies = {}
        self.function_global_resource_dependencies = {}
        self.function_stage_parameter_dependencies = {}
        self.function_stage_output_dependencies = {}
        self.function_metal_wave_lane_dependencies = {}
        self.function_metal_wave_lane_parameter_names = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.current_sampler_parameters = set()
        self.current_sampler_parameter_array_sizes = {}
        self.texture_variable_types = {}
        self.texture_variable_raw_types = {}
        self.current_texture_parameters = {}
        self.current_texture_parameter_raw_types = {}
        self.current_texture_parameter_array_sizes = {}
        self.current_texture_alias_sources = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.function_structured_buffer_length_dependencies = {}
        self.global_structured_buffer_length_dependencies = set()
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function_return_wrapper = None
        self.current_expression_expected_type = None
        self.suppress_image_load_component_suffix = False
        self.current_generic_function_substitutions = {}
        self.local_variable_types = {}
        self.current_address_space_variables = {}
        self.current_local_identifier_remaps = {}
        self.struct_member_types = {}
        self.struct_member_address_spaces = {}
        self.structs_by_name = {}
        self.metal_generated_struct_names = set()
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.struct_constructor_uses_braces = True
        self.glsl_buffer_block_struct_names = set()
        self.lowered_glsl_buffer_blocks = {}
        self.required_buffer_atomic_compare_helpers = set()
        self.required_metal_wave_ballot_helper = False
        self.required_metal_wave_match_helper = False
        self.required_metal_wave_mask_contains_helper = False
        self.required_metal_wave_multi_prefix_helpers = set()
        self.required_metal_inverse_helpers = set()
        self.required_metal_ray_query_runtime = False
        self.required_metal_ray_desc_runtime = False
        self.required_metal_ray_query_helpers = set()
        self.current_unused_local_declaration_names = set()
        self.current_metal_wave_lane_index_parameter = None
        self.current_metal_wave_lane_count_parameter = None
        self.current_metal_graphics_builtin_parameter_names = {}
        self.current_metal_compute_builtin_parameter_names = {}
        self.required_glsl_buffer_aggregate_load_helpers = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_glsl_buffer_block_parameters = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.current_metal_mesh_output_config = None
        self.current_metal_mesh_output_parameter = None
        self.current_metal_mesh_grid_properties_parameter = None
        self.current_metal_mesh_payload_parameter = None
        self.current_metal_mesh_payload_type = None
        self.current_metal_mesh_output_accumulators = {}
        self.current_metal_non_thread_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_reasons = {}
        self.current_readonly_raw_buffer_parameters = set()
        self.current_readonly_metal_parameters = set()
        self.current_readonly_metal_parameter_reasons = {}
        self.metal_ray_payload_parameter_types = set()
        self.metal_program_mesh_payload_type = None
        self.function_metal_mesh_dispatch_contexts = {}
        self.function_metal_mesh_dispatch_contexts_by_id = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.glsl_buffer_block_variables = []
        self.metal_temp_variable_index = 0
        self.generated_return_wrapper_struct_names = set()
        self.metal_vertex_entry_input_struct_names = set()
        self.metal_vertex_entry_output_struct_names = set()
        self.metal_fragment_entry_input_struct_names = set()
        self.metal_fragment_entry_output_struct_names = set()
        self.metal_stage_io_struct_names = set()
        self.metal_stage_io_member_lowerings = {}
        self.type_mapping = {
            # Scalar Types
            "void": "void",
            "short": "int",
            "signed short": "int",
            "unsigned short": "uint",
            "int": "int",
            "signed int": "int",
            "unsigned int": "uint",
            "long": "int64_t",
            "signed long": "int64_t",
            "unsigned long": "uint64_t",
            "ulong": "uint64_t",
            "f16": "half",
            "f32": "float",
            "f64": "double",
            "float": "float",
            "half": "half",
            "float16": "half",
            "min16float": "half",
            "min10float": "half",
            "i8": "int",
            "u8": "uint",
            "i16": "int",
            "u16": "uint",
            "i32": "int",
            "u32": "uint",
            "str": "int",
            "char": "int",
            "signed char": "int",
            "int8": "int",
            "int8_t": "int",
            "uchar": "uint",
            "unsigned char": "uint",
            "uint8": "uint",
            "uint8_t": "uint",
            "int16": "int",
            "int16_t": "int",
            "int32_t": "int",
            "int64": "int64_t",
            "int64_t": "int64_t",
            "ptrdiff_t": "int64_t",
            "uint16": "uint",
            "uint16_t": "uint",
            "uint32_t": "uint",
            "uint64": "uint64_t",
            "uint64_t": "uint64_t",
            "size_t": "uint64_t",
            "min16int": "int",
            "min12int": "int",
            "min16uint": "uint",
            "bool": "bool",
            # Vector Types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "vec2<f16>": "half2",
            "vec3<f16>": "half3",
            "vec4<f16>": "half4",
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i8>": "int2",
            "vec3<i8>": "int3",
            "vec4<i8>": "int4",
            "vec2<u8>": "uint2",
            "vec3<u8>": "uint3",
            "vec4<u8>": "uint4",
            "vec2<i16>": "int2",
            "vec3<i16>": "int3",
            "vec4<i16>": "int4",
            "vec2<u16>": "uint2",
            "vec3<u16>": "uint3",
            "vec4<u16>": "uint4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "vec2<bool>": "bool2",
            "vec3<bool>": "bool3",
            "vec4<bool>": "bool4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "short2": "int2",
            "short3": "int3",
            "short4": "int4",
            "ushort2": "uint2",
            "ushort3": "uint3",
            "ushort4": "uint4",
            "char2": "int2",
            "char3": "int3",
            "char4": "int4",
            "uchar2": "uint2",
            "uchar3": "uint3",
            "uchar4": "uint4",
            "packed_int2": "int2",
            "packed_int3": "int3",
            "packed_int4": "int4",
            "simd_int2": "int2",
            "simd_int3": "int3",
            "simd_int4": "int4",
            "packed_uint2": "uint2",
            "packed_uint3": "uint3",
            "packed_uint4": "uint4",
            "simd_uint2": "uint2",
            "simd_uint3": "uint3",
            "simd_uint4": "uint4",
            "int2": "int2",
            "int3": "int3",
            "int4": "int4",
            "uint2": "uint2",
            "uint3": "uint3",
            "uint4": "uint4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "float2": "float2",
            "float3": "float3",
            "float4": "float4",
            "packed_float2": "float2",
            "packed_float3": "float3",
            "packed_float4": "float4",
            "simd_float2": "float2",
            "simd_float3": "float3",
            "simd_float4": "float4",
            "half2": "half2",
            "half3": "half3",
            "half4": "half4",
            "packed_half2": "half2",
            "packed_half3": "half3",
            "packed_half4": "half4",
            "simd_half2": "half2",
            "simd_half3": "half3",
            "simd_half4": "half4",
            "f16vec2": "half2",
            "f16vec3": "half3",
            "f16vec4": "half4",
            "i8vec2": "int2",
            "i8vec3": "int3",
            "i8vec4": "int4",
            "i16vec2": "int2",
            "i16vec3": "int3",
            "i16vec4": "int4",
            "u8vec2": "uint2",
            "u8vec3": "uint3",
            "u8vec4": "uint4",
            "u16vec2": "uint2",
            "u16vec3": "uint3",
            "u16vec4": "uint4",
            "min16float2": "half2",
            "min16float3": "half3",
            "min16float4": "half4",
            "min10float2": "half2",
            "min10float3": "half3",
            "min10float4": "half4",
            "min16int2": "int2",
            "min16int3": "int3",
            "min16int4": "int4",
            "min12int2": "int2",
            "min12int3": "int3",
            "min12int4": "int4",
            "min16uint2": "uint2",
            "min16uint3": "uint3",
            "min16uint4": "uint4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "bool2": "bool2",
            "bool3": "bool3",
            "bool4": "bool4",
            "sampler": "sampler",
            "comparison_sampler": "sampler",
            "sampler1D": "texture1d<float>",
            "sampler1DArray": "texture1d_array<float>",
            "sampler2D": "texture2d<float>",
            "sampler3D": "texture3d<float>",
            "samplerCube": "texturecube<float>",
            "sampler2DArray": "texture2d_array<float>",
            "samplerCubeArray": "texturecube_array<float>",
            "sampler2DMS": "texture2d_ms<float>",
            "sampler2DMSArray": "texture2d_ms_array<float>",
            "sampler2DShadow": "depth2d<float>",
            "sampler2DArrayShadow": "depth2d_array<float>",
            "samplerCubeShadow": "depthcube<float>",
            "samplerCubeArrayShadow": "depthcube_array<float>",
            "isampler1D": "texture1d<int>",
            "isampler1DArray": "texture1d_array<int>",
            "isampler2D": "texture2d<int>",
            "isampler3D": "texture3d<int>",
            "isamplerCube": "texturecube<int>",
            "isampler2DArray": "texture2d_array<int>",
            "isamplerCubeArray": "texturecube_array<int>",
            "isampler2DMS": "texture2d_ms<int>",
            "isampler2DMSArray": "texture2d_ms_array<int>",
            "usampler1D": "texture1d<uint>",
            "usampler1DArray": "texture1d_array<uint>",
            "usampler2D": "texture2d<uint>",
            "usampler3D": "texture3d<uint>",
            "usamplerCube": "texturecube<uint>",
            "usampler2DArray": "texture2d_array<uint>",
            "usamplerCubeArray": "texturecube_array<uint>",
            "usampler2DMS": "texture2d_ms<uint>",
            "usampler2DMSArray": "texture2d_ms_array<uint>",
            "accelerationStructureEXT": "instance_acceleration_structure",
            "RaytracingAccelerationStructure": "instance_acceleration_structure",
            "AccelerationStructure": "instance_acceleration_structure",
            "acceleration_structure": "instance_acceleration_structure",
            "instance_acceleration_structure": "instance_acceleration_structure",
            "primitive_acceleration_structure": "primitive_acceleration_structure",
            "visible_function_table": "visible_function_table",
            "visibleFunctionTable": "visible_function_table",
            "intersection_function_table": "intersection_function_table",
            "intersectionFunctionTable": "intersection_function_table",
            "iimage1D": "texture1d<int, access::read_write>",
            "iimage1DArray": "texture1d_array<int, access::read_write>",
            "iimage2D": "texture2d<int, access::read_write>",
            "iimage3D": "texture3d<int, access::read_write>",
            "iimage2DArray": "texture2d_array<int, access::read_write>",
            "iimage2DMS": "texture2d_ms<int, access::read>",
            "iimage2DMSArray": "texture2d_ms_array<int, access::read>",
            "uimage1D": "texture1d<uint, access::read_write>",
            "uimage1DArray": "texture1d_array<uint, access::read_write>",
            "uimage2D": "texture2d<uint, access::read_write>",
            "uimage3D": "texture3d<uint, access::read_write>",
            "uimage2DArray": "texture2d_array<uint, access::read_write>",
            "uimage2DMS": "texture2d_ms<uint, access::read>",
            "uimage2DMSArray": "texture2d_ms_array<uint, access::read>",
            "image1D": "texture1d<float, access::read_write>",
            "image1DArray": "texture1d_array<float, access::read_write>",
            "image2D": "texture2d<float, access::read_write>",
            "image3D": "texture3d<float, access::read_write>",
            "imageCube": "texture2d_array<float, access::read_write>",
            "image2DArray": "texture2d_array<float, access::read_write>",
            "image2DMS": "texture2d_ms<float, access::read>",
            "image2DMSArray": "texture2d_ms_array<float, access::read>",
            # Matrix Types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float3x2",
            "mat2x4": "float4x2",
            "mat3x2": "float2x3",
            "mat3x3": "float3x3",
            "mat3x4": "float4x3",
            "mat4x2": "float2x4",
            "mat4x3": "float3x4",
            "mat4x4": "float4x4",
            "half2x2": "half2x2",
            "half2x3": "half2x3",
            "half2x4": "half2x4",
            "half3x2": "half3x2",
            "half3x3": "half3x3",
            "half3x4": "half3x4",
            "half4x2": "half4x2",
            "half4x3": "half4x3",
            "half4x4": "half4x4",
            "f16mat2": "half2x2",
            "f16mat3": "half3x3",
            "f16mat4": "half4x4",
            "f16mat2x2": "half2x2",
            "f16mat2x3": "half3x2",
            "f16mat2x4": "half4x2",
            "f16mat3x2": "half2x3",
            "f16mat3x3": "half3x3",
            "f16mat3x4": "half4x3",
            "f16mat4x2": "half2x4",
            "f16mat4x3": "half3x4",
            "f16mat4x4": "half4x4",
            "simd_float2x2": "float2x2",
            "simd_float2x3": "float2x3",
            "simd_float2x4": "float2x4",
            "simd_float3x2": "float3x2",
            "simd_float3x3": "float3x3",
            "simd_float3x4": "float3x4",
            "simd_float4x2": "float4x2",
            "simd_float4x3": "float4x3",
            "simd_float4x4": "float4x4",
            "simd_half2x2": "half2x2",
            "simd_half2x3": "half2x3",
            "simd_half2x4": "half2x4",
            "simd_half3x2": "half3x2",
            "simd_half3x3": "half3x3",
            "simd_half3x4": "half3x4",
            "simd_half4x2": "half4x2",
            "simd_half4x3": "half4x3",
            "simd_half4x4": "half4x4",
            "min16float2x2": "half2x2",
            "min16float2x3": "half2x3",
            "min16float2x4": "half2x4",
            "min16float3x2": "half3x2",
            "min16float3x3": "half3x3",
            "min16float3x4": "half3x4",
            "min16float4x2": "half4x2",
            "min16float4x3": "half4x3",
            "min16float4x4": "half4x4",
            "min10float2x2": "half2x2",
            "min10float2x3": "half2x3",
            "min10float2x4": "half2x4",
            "min10float3x2": "half3x2",
            "min10float3x3": "half3x3",
            "min10float3x4": "half3x4",
            "min10float4x2": "half4x2",
            "min10float4x3": "half4x3",
            "min10float4x4": "half4x4",
        }

        self.semantic_map = {
            # Vertex inputs
            "gl_VertexIndex": "vertex_id",
            "gl_VertexID": "vertex_id",
            "SV_VertexID": "vertex_id",
            "SV_VertexId": "vertex_id",
            "sv_vertex_id": "vertex_id",
            "sv_vertexid": "vertex_id",
            "gl_InstanceID": "instance_id",
            "SV_InstanceID": "instance_id",
            "SV_InstanceId": "instance_id",
            "sv_instance_id": "instance_id",
            "sv_instanceid": "instance_id",
            "gl_IsFrontFace": "is_front_facing",
            "SV_IsFrontFace": "is_front_facing",
            "SV_IsFrontFacing": "is_front_facing",
            "sv_is_front_face": "is_front_facing",
            "sv_isfrontface": "is_front_facing",
            "gl_PrimitiveID": "primitive_id",
            "SV_PrimitiveID": "primitive_id",
            "SV_PrimitiveId": "primitive_id",
            "sv_primitive_id": "primitive_id",
            "sv_primitiveid": "primitive_id",
            "Position": "attribute(0)",
            "POSITION": "attribute(0)",
            "Normal": "attribute(1)",
            "NORMAL": "attribute(1)",
            "Tangent": "attribute(2)",
            "TANGENT": "attribute(2)",
            "Binormal": "attribute(3)",
            "BINORMAL": "attribute(3)",
            "TexCoord": "attribute(4)",
            "TEXCOORD": "attribute(4)",
            "TexCoord0": "attribute(5)",
            "TEXCOORD0": "attribute(5)",
            "TexCoord1": "attribute(6)",
            "TEXCOORD1": "attribute(6)",
            "TexCoord2": "attribute(7)",
            "TEXCOORD2": "attribute(7)",
            "TexCoord3": "attribute(8)",
            "TEXCOORD3": "attribute(8)",
            "TexCoord4": "attribute(9)",
            "TEXCOORD4": "attribute(9)",
            "TexCoord5": "attribute(10)",
            "TEXCOORD5": "attribute(10)",
            "TexCoord6": "attribute(11)",
            "TEXCOORD6": "attribute(11)",
            "TexCoord7": "attribute(12)",
            "TEXCOORD7": "attribute(12)",
            # Vertex outputs
            "gl_Position": "position",
            "SV_POSITION": "position",
            "SV_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            # Fragment inputs
            "gl_FragColor": "[[color(0)]]",
            "gl_FragColor0": "[[color(0)]]",
            "gl_FragColor1": "[[color(1)]]",
            "gl_FragColor2": "[[color(2)]]",
            "gl_FragColor3": "[[color(3)]]",
            "gl_FragColor4": "[[color(4)]]",
            "gl_FragColor5": "[[color(5)]]",
            "gl_FragColor6": "[[color(6)]]",
            "gl_FragColor7": "[[color(7)]]",
            "gl_FragDepth": "depth(any)",
            "gl_FragStencilRefEXT": "stencil_ref",
            "SV_TARGET": "color(0)",
            "SV_TARGET0": "color(0)",
            "SV_TARGET1": "color(1)",
            "SV_TARGET2": "color(2)",
            "SV_TARGET3": "color(3)",
            "SV_TARGET4": "color(4)",
            "SV_TARGET5": "color(5)",
            "SV_TARGET6": "color(6)",
            "SV_TARGET7": "color(7)",
            "SV_Target": "color(0)",
            "SV_Target0": "color(0)",
            "SV_Target1": "color(1)",
            "SV_Target2": "color(2)",
            "SV_Target3": "color(3)",
            "SV_Target4": "color(4)",
            "SV_Target5": "color(5)",
            "SV_Target6": "color(6)",
            "SV_Target7": "color(7)",
            "SV_DEPTH": "depth(any)",
            "SV_Depth": "depth(any)",
            # Additional Metal-specific attributes
            "gl_FragCoord": "position",
            "gl_BaryCoordEXT": "barycentric_coord",
            "gl_BaryCoordNoPerspEXT": "barycentric_coord",
            "gl_FrontFacing": "is_front_facing",
            "gl_PointCoord": "point_coord",
            "gl_SampleID": "sample_id",
            "SV_SampleIndex": "sample_id",
            "gl_SampleMask": "sample_mask",
            "gl_SampleMaskIn": "sample_mask",
            "SV_Coverage": "sample_mask",
            # Compute shader specific
            "gl_GlobalInvocationID": "thread_position_in_grid",
            "SV_DispatchThreadID": "thread_position_in_grid",
            "SV_DispatchThreadId": "thread_position_in_grid",
            "sv_dispatch_thread_id": "thread_position_in_grid",
            "sv_dispatchthreadid": "thread_position_in_grid",
            "gl_LocalInvocationID": "thread_position_in_threadgroup",
            "SV_GroupThreadID": "thread_position_in_threadgroup",
            "SV_GroupThreadId": "thread_position_in_threadgroup",
            "sv_group_thread_id": "thread_position_in_threadgroup",
            "sv_groupthreadid": "thread_position_in_threadgroup",
            "gl_WorkGroupID": "threadgroup_position_in_grid",
            "SV_GroupID": "threadgroup_position_in_grid",
            "SV_GroupId": "threadgroup_position_in_grid",
            "sv_group_id": "threadgroup_position_in_grid",
            "sv_groupid": "threadgroup_position_in_grid",
            "gl_LocalInvocationIndex": "thread_index_in_threadgroup",
            "SV_GroupIndex": "thread_index_in_threadgroup",
            "sv_group_index": "thread_index_in_threadgroup",
            "sv_groupindex": "thread_index_in_threadgroup",
            "gl_WorkGroupSize": "threads_per_threadgroup",
            "gl_NumWorkGroups": "threadgroups_per_grid",
            "gl_SubgroupInvocationID": "thread_index_in_simdgroup",
            "gl_SubgroupSize": "threads_per_simdgroup",
            # Ray tracing / payload semantics
            "payload": "payload",
            "mesh_payload": "payload",
            "hlsl_mesh_payload": "payload",
            "task_payload": "payload",
            "taskPayloadSharedEXT": "payload",
            "hit_attribute": "hit_attribute",
            "callable_data": "callable_data",
            "shader_record": "shader_record",
        }

    def generate(self, ast):
        """Generate complete Metal Shading Language source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_stage(self, ast, shader_type):
        """Generate Metal source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def generate_program(self, ast, target_stage=None):
        """Render an AST to Metal, optionally filtering stage entry points."""
        target_stage = normalize_stage_name(target_stage)
        self.validate_supported_stage_types(ast, target_stage)

        self.texture_variables = []
        self.acceleration_structure_variables = []
        self.visible_function_table_variables = []
        self.intersection_function_table_variables = []
        self.unsupported_metal_acceleration_structure_array_variables = {}
        self.unsupported_metal_ray_function_table_array_variables = {}
        self.sampler_variables = []
        self.structured_buffer_variables = []
        self.structured_buffer_length_variables = []
        self.structured_buffer_counter_variables = []
        self.metal_buffer_resource_variables = []
        self.metal_resource_binding_indices_by_id = {}
        self.metal_program_scope_value_globals = set()
        self.metal_program_scope_groupshared_globals = set()
        self.metal_lowered_program_scope_groupshared_globals_by_function = {}
        self.metal_lowered_program_scope_groupshared_global_ids = set()
        self.metal_program_scope_value_global_types = {}
        self.glsl_buffer_block_variables = []
        self.lowered_glsl_buffer_blocks = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.metal_temp_variable_index = 0
        self.generated_return_wrapper_struct_names = set()
        self.metal_vertex_entry_input_struct_names = set()
        self.metal_vertex_entry_output_struct_names = set()
        self.metal_fragment_entry_input_struct_names = set()
        self.metal_fragment_entry_output_struct_names = set()
        self.metal_stage_io_struct_names = set()
        self.metal_stage_io_member_lowerings = {}
        self.hlsl_program_constant_global_ids = set()
        self.cbuffer_variables = list(getattr(ast, "cbuffers", []) or [])
        hlsl_program_constants = self.synthetic_hlsl_program_constants_cbuffer(ast)
        if hlsl_program_constants is not None:
            self.cbuffer_variables.append(hlsl_program_constants)
        self.cbuffer_binding_indices = {}
        self.cbuffers_by_name = {
            cbuffer.name: cbuffer
            for cbuffer in self.cbuffer_variables
            if getattr(cbuffer, "name", None)
        }
        all_functions = self.all_functions(ast)
        self.validate_metal_mesh_payload_parameter_placement(ast, all_functions)
        self.user_function_names = {
            func.name for func in all_functions if getattr(func, "name", None)
        }
        self.metal_atomic_fence_calls = self.collect_metal_atomic_fence_calls(ast)
        self.requires_metal_atomic_fence = bool(self.metal_atomic_fence_calls)
        self.requires_metal_system_thread_scope = any(
            self.atomic_fence_operand_identifier(call.args[2]) == "thread_scope_system"
            for call in self.metal_atomic_fence_calls
            if len(call.args) == 3
        )
        self.metal_resource_memory_contracts = (
            self.collect_metal_resource_memory_contracts(ast)
        )
        self.requires_metal_resource_coherence = any(
            kind == "coherent" for kind, _scope in self.metal_resource_memory_contracts
        )
        self.functions_by_name = {
            func.name: func for func in all_functions if getattr(func, "name", None)
        }
        self.function_parameter_infos = self.collect_function_parameter_infos(
            all_functions
        )
        self.function_parameter_nodes = self.collect_function_parameter_nodes(
            all_functions
        )
        self.metal_skipped_function_parameter_indices = (
            self.collect_unused_array_parameter_indices(all_functions)
        )
        self.function_return_types = {
            func.name: self.type_name_string(getattr(func, "return_type", "void"))
            for func in all_functions
            if getattr(func, "name", None)
        }
        self.function_parameter_names = collect_function_parameter_names(all_functions)
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                all_functions,
                self.function_parameter_names,
                self.iter_ast_nodes,
                self.function_call_name,
                self.expression_name,
            )
        )
        self.function_cbuffer_dependencies = self.collect_function_cbuffer_dependencies(
            all_functions
        )
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            self.cbuffer_variables
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            self.cbuffer_variables
        )
        if not self.cbuffer_variables:
            self.ambiguous_cbuffer_members = set()
        self.current_sampler_parameters = set()
        self.current_sampler_parameter_array_sizes = {}
        self.texture_variable_types = {}
        self.texture_variable_raw_types = {}
        self.current_texture_parameters = {}
        self.current_texture_parameter_raw_types = {}
        self.current_texture_parameter_array_sizes = {}
        self.current_texture_alias_sources = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.function_structured_buffer_length_dependencies = {}
        self.global_structured_buffer_length_dependencies = set()
        self.function_global_resource_dependencies = {}
        self.function_stage_parameter_dependencies = {}
        self.function_metal_wave_lane_dependencies = {}
        self.function_metal_wave_lane_parameter_names = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.required_image_atomic_compare_helpers = set()
        self.required_metal_inverse_helpers = set()
        self.required_metal_ray_query_runtime = False
        self.required_metal_ray_desc_runtime = False
        self.required_metal_ray_query_helpers = set()
        self.required_glsl_buffer_aggregate_load_helpers = {}
        self.current_glsl_buffer_block_parameters = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.current_metal_mesh_output_config = None
        self.current_metal_mesh_output_parameter = None
        self.current_metal_mesh_grid_properties_parameter = None
        self.current_metal_mesh_payload_parameter = None
        self.current_metal_mesh_payload_type = None
        self.current_metal_mesh_output_accumulators = {}
        self.current_metal_non_thread_payload_parameters = set()
        self.metal_ray_payload_parameter_types = set()
        self.function_metal_mesh_dispatch_contexts = {}
        self.function_metal_mesh_dispatch_contexts_by_id = {}
        self.metal_program_mesh_payload_type = self.metal_mesh_payload_type_for_program(
            ast
        )
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        structs = deduplicate_named_declarations(
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            "struct",
        )
        self.structs_by_name = {
            node.name: node for node in structs if isinstance(node, StructNode)
        }
        self.metal_generated_struct_names = set()
        self.metal_vertex_entry_input_struct_names = (
            self.collect_metal_stage_entry_parameter_struct_names(
                ast, target_stage, "vertex"
            )
        )
        self.metal_vertex_entry_output_struct_names = (
            self.collect_metal_stage_entry_return_struct_names(
                ast, target_stage, "vertex"
            )
        )
        self.metal_fragment_entry_input_struct_names = (
            self.collect_metal_stage_entry_parameter_struct_names(
                ast, target_stage, "fragment"
            )
        )
        self.metal_fragment_entry_output_struct_names = (
            self.collect_metal_stage_entry_return_struct_names(
                ast, target_stage, "fragment"
            )
        )
        self.metal_stage_io_struct_names = (
            set(self.metal_vertex_entry_input_struct_names)
            | set(self.metal_vertex_entry_output_struct_names)
            | set(self.metal_fragment_entry_input_struct_names)
            | set(self.metal_fragment_entry_output_struct_names)
        )
        self.metal_ray_payload_parameter_types = (
            self.collect_metal_ray_payload_parameter_types(getattr(ast, "stages", {}))
        )
        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            structs
        )
        self.generic_struct_definitions = collect_generic_struct_definitions(
            structs,
            excluded_names=set(self.generic_enum_struct_definitions),
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )
        self.generic_struct_specializations = collect_generic_struct_specializations(
            ast,
            self.generic_struct_definitions,
            self.type_name_string,
        )
        self.plain_enums = collect_plain_enums(structs)
        self.struct_payload_enums = collect_struct_payload_enums(structs)
        self.enum_type_names = collect_enum_type_names(self.plain_enums)
        self.enum_struct_type_names = (
            collect_enum_type_names(self.struct_payload_enums)
            | set(self.generic_enum_struct_definitions)
            | {
                specialization["struct_name"]
                for specialization in self.generic_enum_specializations.values()
            }
        )
        self.enum_struct_variant_fields = collect_enum_struct_variant_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constructors = collect_enum_variant_constructors(
            self.struct_payload_enums
        )
        self.enum_variant_constructor_fields = collect_enum_variant_constructor_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constants = {
            **collect_enum_variant_constants(
                self.plain_enums + self.struct_payload_enums
            ),
            **collect_generic_enum_variant_constants(
                self.generic_enum_struct_definitions
            ),
        }
        stage_local_resource_variables = collect_stage_local_variables(
            ast, target_stage, self.is_stage_local_resource_variable
        )
        self.metal_source_binding_stage_by_id = (
            self.collect_stage_local_resource_stage_scopes(
                ast, target_stage, stage_local_resource_variables
            )
        )
        global_vars = deduplicate_named_declarations(
            list(getattr(ast, "global_variables", []) or [])
            + stage_local_resource_variables,
            "Metal resource",
        )
        self.metal_lowered_program_scope_groupshared_globals_by_function = (
            self.collect_metal_lowered_program_scope_groupshared_globals(
                ast, target_stage, global_vars, all_functions
            )
        )
        self.metal_lowered_program_scope_groupshared_global_ids = {
            id(node)
            for nodes in (
                self.metal_lowered_program_scope_groupshared_globals_by_function.values()
            )
            for node in nodes
        }
        self.metal_spirv_interface_variables = [
            node for node in global_vars if self.is_spirv_stage_interface_layout(node)
        ]
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(
                list(global_vars) + self.collect_function_parameters(all_functions)
            )
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.function_structured_buffer_length_dependencies = (
            self.collect_function_structured_buffer_length_dependencies(all_functions)
        )
        self.global_structured_buffer_length_dependencies = (
            self.collect_global_structured_buffer_length_dependencies(
                all_functions, global_vars
            )
        )
        self.validate_global_resource_shadows(ast)
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function_return_wrapper = None
        self.current_expression_expected_type = None
        self.suppress_image_load_component_suffix = False
        self.required_buffer_atomic_compare_helpers = set()
        self.required_metal_wave_ballot_helper = False
        self.required_metal_wave_match_helper = False
        self.required_metal_wave_mask_contains_helper = False
        self.required_metal_wave_multi_prefix_helpers = set()
        self.required_metal_inverse_helpers = set()
        self.required_metal_ray_query_runtime = False
        self.required_metal_ray_desc_runtime = False
        self.required_metal_ray_query_helpers = set()
        self.local_variable_types = {}
        self.current_metal_graphics_builtin_parameter_names = {}
        (
            self.lowered_glsl_buffer_blocks,
            self.glsl_buffer_block_lowering_failures,
            self.glsl_buffer_block_struct_lowering_failures,
        ) = collect_lowered_glsl_buffer_blocks(
            global_vars,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: self.literal_int_value(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="metal_type",
            unsupported_type_message=(
                "type is not supported by Metal pointer/offset lowering"
            ),
        )
        self.lowered_glsl_buffer_block_struct_names = {
            block["type_name"] for block in self.lowered_glsl_buffer_blocks.values()
        }
        self.unsupported_glsl_buffer_block_struct_names = set(
            self.glsl_buffer_block_struct_lowering_failures
        )
        _, _, parameter_struct_failures = (
            self.collect_lowered_glsl_buffer_block_parameters(
                self.collect_function_parameters(all_functions)
            )
        )
        self.unsupported_glsl_buffer_block_struct_names.update(
            parameter_struct_failures
        )
        self.unsupported_glsl_buffer_block_functions = (
            self.collect_unsupported_glsl_buffer_block_functions(all_functions)
        )
        self.struct_member_types = collect_struct_member_types(
            structs, self.type_name_string
        )
        self.struct_member_address_spaces = self.collect_struct_member_address_spaces(
            structs
        )
        self.struct_member_types.update(
            collect_generic_enum_specialization_member_types(
                self,
                self.generic_enum_specializations,
            )
        )
        self.struct_member_types.update(
            collect_generic_struct_specialization_member_types(
                self,
                self.generic_struct_specializations,
            )
        )
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            all_functions,
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        self.user_function_names.update(
            func.name for func in generic_function_specializations.values()
        )
        if generic_function_specializations:
            self.function_parameter_names.update(
                collect_function_parameter_names(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_nodes.update(
                self.collect_function_parameter_nodes(
                    generic_function_specializations.values()
                )
            )
            generic_specialization_roots = [
                ast,
                *generic_function_specializations.values(),
            ]
            self.generic_enum_specializations = collect_generic_enum_specializations(
                generic_specialization_roots,
                self.generic_enum_struct_definitions,
                self.type_name_string,
            )
            self.generic_struct_specializations = (
                collect_generic_struct_specializations(
                    generic_specialization_roots,
                    self.generic_struct_definitions,
                    self.type_name_string,
                )
            )
            for func in generic_function_specializations.values():
                self.add_metal_generic_specializations_from_type_text(
                    self.type_name_string(getattr(func, "return_type", None))
                )
                for parameter in getattr(
                    func, "parameters", getattr(func, "params", [])
                ):
                    self.add_metal_generic_specializations_from_type_text(
                        self.type_name_string(getattr(parameter, "param_type", None))
                    )
            self.enum_struct_type_names = (
                collect_enum_type_names(self.struct_payload_enums)
                | set(self.generic_enum_struct_definitions)
                | {
                    specialization["struct_name"]
                    for specialization in self.generic_enum_specializations.values()
                }
            )
            self.enum_variant_constants = {
                **collect_enum_variant_constants(
                    self.plain_enums + self.struct_payload_enums
                ),
                **collect_generic_enum_variant_constants(
                    self.generic_enum_struct_definitions
                ),
            }
            self.struct_member_types.update(
                collect_generic_enum_specialization_member_types(
                    self,
                    self.generic_enum_specializations,
                )
            )
            self.struct_member_types.update(
                collect_generic_struct_specialization_member_types(
                    self,
                    self.generic_struct_specializations,
                )
            )
        wave_dependency_functions = list(all_functions) + list(
            generic_function_specializations.values()
        )
        self.function_metal_wave_lane_dependencies = (
            self.collect_function_metal_wave_lane_dependencies(
                wave_dependency_functions
            )
        )
        self.function_metal_wave_lane_parameter_names = (
            self.collect_function_metal_wave_lane_parameter_names(
                wave_dependency_functions
            )
        )
        self.function_metal_mesh_dispatch_contexts = (
            self.collect_function_metal_mesh_dispatch_contexts(
                list(all_functions) + list(generic_function_specializations.values())
            )
        )
        self.metal_stage_io_member_lowerings = (
            self.collect_metal_stage_io_member_lowerings(structs)
        )
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        pre_lines = []
        for directive in preprocessors:
            line = self.generate_preprocessor_directive(directive)
            if line:
                pre_lines.append(line)
        if (
            self.requires_metal_system_thread_scope
            or self.requires_metal_resource_coherence
        ) and not any("#pragma metal internals" in line.lower() for line in pre_lines):
            code += "#pragma METAL internals : enable\n"
        if pre_lines:
            code += "\n".join(pre_lines) + "\n"
        if not any("metal_stdlib" in line for line in pre_lines):
            code += "#include <metal_stdlib>\n"
        if self.requires_metal_atomic_fence and not any(
            "metal_atomic" in line for line in pre_lines
        ):
            code += "#include <metal_atomic>\n"
        if self.uses_cooperative_matrix(ast) and not any(
            "metal_simdgroup_matrix" in line for line in pre_lines
        ):
            code += "#include <metal_simdgroup_matrix>\n"
        code += "using namespace metal;\n"
        if self.uses_metal_raytracing_namespace(ast, global_vars, all_functions):
            code += "using namespace metal::raytracing;\n"
        code += "\n"
        if self.requires_metal_system_thread_scope:
            code += self.generate_metal_system_thread_scope_support()
        if self.has_geometry_stage(ast, target_stage):
            code += self.generate_metal_geometry_stream_helpers()
        if self.has_tessellation_stage(ast, target_stage):
            code += self.generate_metal_tessellation_patch_helpers()
        code += generate_enum_constants(
            self,
            self.plain_enums + self.struct_payload_enums,
            qualifier=self.metal_unused_declaration_qualifier("constant"),
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
            qualifier=self.metal_unused_declaration_qualifier("constant"),
        )
        leading_constants, struct_dependent_constants = (
            partition_constants_by_struct_dependency(
                getattr(ast, "constants", []) or [], structs
            )
        )
        code += self.generate_constants(ast, leading_constants)
        code += self.generate_metal_builtin_limit_fallbacks(ast, all_functions)
        code += self.generate_metal_struct_declarations(structs)
        code += self.generate_constants(
            ast, struct_dependent_constants, include_function_constants=False
        )
        code += self.generate_metal_stage_io_array_helpers()
        code += self.generate_metal_enum_constructor_functions(
            self.struct_payload_enums
        )
        code += self.generate_metal_generic_enum_constructor_functions(
            self.generic_enum_specializations
        )

        texture_register = 0
        sampler_register = 0
        buffer_register = 0
        used_resource_bindings = {}
        source_resource_bindings = {}
        self.pre_reserve_cbuffer_bindings(
            used_resource_bindings, source_resource_bindings
        )
        self.reserve_explicit_global_resource_bindings(
            global_vars, used_resource_bindings, source_resource_bindings
        )
        buffer_register = self.reserve_cbuffer_bindings(
            used_resource_bindings,
            source_resource_bindings,
        )
        for i, node in enumerate(global_vars):
            if self.is_spirv_stage_interface_layout(node):
                continue

            resource_count = 1
            array_size = None
            if hasattr(node, "var_type"):
                if (
                    hasattr(node.var_type, "name")
                    or hasattr(node.var_type, "element_type")
                    or isinstance(node.var_type, (PointerType, ReferenceType))
                ):
                    if (
                        hasattr(node.var_type, "element_type")
                        and str(type(node.var_type)).find("ArrayType") != -1
                    ):  # ArrayType
                        base_type = self.convert_type_node_to_string(
                            node.var_type.element_type
                        )
                        array_size = (
                            self.expression_to_string(node.var_type.size)
                            if node.var_type.size
                            else self.resource_array_size_hints.get(node.name, "")
                        )
                        vtype = base_type
                        array_suffix = f"[{array_size}]" if array_size else "[]"
                        resource_count = self.resource_array_count(
                            node.var_type.size if node.var_type.size else array_size
                        )
                    else:
                        vtype = self.convert_type_node_to_string(node.var_type)
                        array_suffix = ""
                else:
                    vtype = str(node.var_type)
                    array_suffix = ""
            elif hasattr(node, "vtype"):
                vtype = node.vtype
                array_suffix = ""
            else:
                vtype = "float"
                array_suffix = ""

            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if id(node) in self.hlsl_program_constant_global_ids:
                continue
            if id(node) in self.metal_lowered_program_scope_groupshared_global_ids:
                continue
            if self.is_metal_function_constant_variable(node):
                continue

            if self.is_metal_argument_buffer_global(node):
                binding = self.metal_argument_buffer_binding(node)
                declaration = self.format_metal_argument_buffer_global(
                    node, vtype, var_name
                )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                code += declaration
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
            if lowered_block is not None:
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                self.glsl_buffer_block_variables.append(
                    (node, binding, lowered_block, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_glsl_buffer_block_variable(node, vtype):
                code += self.glsl_buffer_block_diagnostic(
                    "Metal", vtype, var_name, node
                )
                self.unsupported_glsl_buffer_block_variables.add(var_name)
                self.unsupported_glsl_buffer_block_variable_types[var_name] = (
                    self.type_name_string(vtype)
                )
                code += self.unsupported_glsl_buffer_block_variable_placeholder(
                    "Metal", vtype, var_name
                )
                continue

            metal_buffer_address_space = self.metal_buffer_resource_address_space(
                node, vtype
            )
            if metal_buffer_address_space is not None:
                binding_count = self.metal_buffer_resource_binding_count(
                    node, array_size
                )
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        binding_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    binding_count,
                    var_name,
                    node,
                    buffer_register,
                )
                self.metal_buffer_resource_variables.append(
                    (node, binding, vtype, array_size, metal_buffer_address_space)
                )
                buffer_register = max(buffer_register, binding + binding_count)
                continue

            if self.is_acceleration_structure_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                mapped_type = self.map_resource_type_with_format(vtype, node)
                if array_size is not None:
                    self.unsupported_metal_acceleration_structure_array_variables[
                        var_name
                    ] = mapped_type
                    code += (
                        self.unsupported_metal_acceleration_structure_array_diagnostic(
                            var_name
                        )
                    )
                    buffer_register = max(buffer_register, binding + resource_count)
                    continue
                self.acceleration_structure_variables.append(
                    (node, binding, mapped_type, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_visible_function_table_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                if array_size is not None:
                    self.unsupported_metal_ray_function_table_array_variables[
                        var_name
                    ] = "visible_function_table"
                    code += self.unsupported_metal_ray_function_table_array_diagnostic(
                        "visible_function_table", var_name
                    )
                    buffer_register = max(buffer_register, binding + resource_count)
                    continue
                mapped_type = self.map_visible_function_table_type(vtype)
                self.visible_function_table_variables.append(
                    (node, binding, mapped_type, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_intersection_function_table_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                if array_size is not None:
                    self.unsupported_metal_ray_function_table_array_variables[
                        var_name
                    ] = "intersection_function_table"
                    code += self.unsupported_metal_ray_function_table_array_diagnostic(
                        "intersection_function_table", var_name
                    )
                    buffer_register = max(buffer_register, binding + resource_count)
                    continue
                mapped_type = self.map_intersection_function_table_type(vtype)
                self.intersection_function_table_variables.append(
                    (node, binding, mapped_type, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_structured_buffer_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                self.structured_buffer_variables.append(
                    (node, binding, vtype, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                if self.structured_buffer_requires_length(var_name):
                    length_binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                    length_name = self.structured_buffer_length_resource_name(var_name)
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "Metal",
                        "buffer",
                        length_binding,
                        resource_count,
                        length_name,
                    )
                    self.structured_buffer_length_variables.append(
                        (node, length_binding, vtype, array_size)
                    )
                    buffer_register = max(
                        buffer_register, length_binding + resource_count
                    )
                if self.structured_buffer_requires_counter(vtype):
                    counter_binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                    counter_name = self.structured_buffer_counter_resource_name(
                        var_name
                    )
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "Metal",
                        "buffer",
                        counter_binding,
                        resource_count,
                        counter_name,
                    )
                    self.structured_buffer_counter_variables.append(
                        (node, counter_binding, vtype, array_size)
                    )
                    buffer_register = max(
                        buffer_register, counter_binding + resource_count
                    )
            elif vtype in [
                "sampler1D",
                "sampler1DArray",
                "sampler2D",
                "sampler3D",
                "samplerCube",
                "sampler2DArray",
                "samplerCubeArray",
                "sampler2DMS",
                "sampler2DMSArray",
                "sampler2DShadow",
                "sampler2DArrayShadow",
                "samplerCubeShadow",
                "samplerCubeArrayShadow",
                "isampler1D",
                "isampler1DArray",
                "isampler2D",
                "isampler3D",
                "isamplerCube",
                "isampler2DArray",
                "isamplerCubeArray",
                "isampler2DMS",
                "isampler2DMSArray",
                "usampler1D",
                "usampler1DArray",
                "usampler2D",
                "usampler3D",
                "usamplerCube",
                "usampler2DArray",
                "usamplerCubeArray",
                "usampler2DMS",
                "usampler2DMSArray",
                "iimage1D",
                "iimage1DArray",
                "iimage2D",
                "iimage3D",
                "iimage2DArray",
                "iimage2DMS",
                "iimage2DMSArray",
                "uimage1D",
                "uimage1DArray",
                "uimage2D",
                "uimage3D",
                "uimage2DArray",
                "uimage2DMS",
                "uimage2DMSArray",
                "image1D",
                "image1DArray",
                "image2D",
                "image3D",
                "imageCube",
                "image2DArray",
                "image2DMS",
                "image2DMSArray",
            ]:
                mapped_type = self.map_resource_type_with_format(vtype, node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "texture"}, ("t", "u")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "texture",
                        texture_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "texture",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    texture_register,
                )
                self.texture_variables.append((node, binding, mapped_type, array_size))
                self.texture_variable_types[node.name] = mapped_type
                self.texture_variable_raw_types[node.name] = self.resource_base_type(
                    vtype
                )
                record_explicit_image_metadata(
                    node.name,
                    node,
                    self.attribute_value_to_string,
                    image_formats=self.image_variable_formats,
                )
                texture_register = max(texture_register, binding + resource_count)
            elif self.resource_base_type(vtype) in {"sampler", "comparison_sampler"}:
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "sampler"}, ("s",)
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "sampler",
                        sampler_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "sampler",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    sampler_register,
                )
                self.sampler_variables.append((node, binding, array_size))
                sampler_register = max(sampler_register, binding + resource_count)
            elif self.is_input_attachment_type_name(vtype):
                code += (
                    "/* unsupported Metal input attachment declaration: "
                    f"'{var_name}' uses {self.type_name_string(vtype)}; "
                    "subpassInput resources require Vulkan subpass lowering */\n"
                )
            else:
                mapped_type = self.map_type(vtype)
                declaration = format_c_style_array_declaration(mapped_type, var_name)
                if array_suffix:
                    declaration = f"{declaration}{array_suffix}"
                qualifier = self.global_variable_qualifier(node)
                declaration = f"{qualifier}{declaration}"
                self.record_metal_program_scope_value_global(
                    var_name, vtype, qualifier, node
                )
                initial_value = getattr(node, "initial_value", None)
                if initial_value is not None:
                    expected_type = getattr(node, "var_type", vtype)
                    init_expr = self.generate_expression_with_expected(
                        initial_value, expected_type
                    )
                    code += f"{declaration} = {init_expr};\n"
                elif self.global_value_variable_requires_initializer(qualifier):
                    diagnostic = self.metal_program_scope_global_initializer_diagnostic(
                        var_name, node
                    )
                    code += f"{diagnostic}\n"
                    init_expr = self.metal_program_scope_global_default_initializer(
                        mapped_type, array_suffix
                    )
                    code += f"{declaration} = {init_expr};\n"
                else:
                    code += f"{declaration};\n"

        self.function_global_resource_dependencies = (
            self.collect_function_global_resource_dependencies(all_functions)
        )
        self.function_stage_parameter_dependencies = (
            self.collect_function_stage_parameter_dependencies(ast, target_stage)
        )
        self.function_stage_output_dependencies = (
            self.collect_function_stage_output_dependencies(ast, target_stage)
        )

        cbuffers = self.cbuffer_variables
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        stage_entry_names = self.stage_entry_names(ast, target_stage)

        functions = getattr(ast, "functions", [])
        functions_code = ""
        for func in functions:
            qualifier_name = function_stage_name(func)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            if generic_function_parameters(func):
                for specialized_func in generic_function_emission_list(self, func):
                    functions_code += self.generate_function(specialized_func)
                continue

            if qualifier_name == "vertex":
                functions_code += "// Vertex Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "fragment":
                functions_code += "// Fragment Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "compute":
                functions_code += "// Compute Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="compute",
                    entry_name=stage_entry_names.get(id(func)),
                    stage_local_variables=self.metal_stage_local_variables_with_lowered_program_groupshared(
                        func
                    ),
                )
            else:
                functions_code += self.generate_function(func)

        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                for func in order_functions_by_dependencies(
                    getattr(stage, "local_functions", []) or [],
                    self.iter_ast_nodes,
                    self.function_call_name,
                    FunctionCallNode,
                ):
                    if generic_function_parameters(func):
                        for specialized_func in generic_function_emission_list(
                            self,
                            func,
                        ):
                            functions_code += self.generate_function(specialized_func)
                    else:
                        functions_code += self.generate_function(func)

                if hasattr(stage, "entry_point"):
                    functions_code += f"// {stage_name.title()} Shader\n"
                    functions_code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                        entry_name=stage_entry_names.get(id(stage.entry_point)),
                        stage_local_variables=self.metal_stage_local_variables_with_lowered_program_groupshared(
                            stage.entry_point,
                            getattr(stage, "local_variables", []),
                        ),
                        stage_node=stage,
                    )

        code += self.generate_image_atomic_compare_helpers()
        code += self.generate_buffer_atomic_compare_helpers()
        code += self.generate_metal_wave_helpers()
        code += self.generate_metal_ray_query_helpers()
        code += self.generate_glsl_buffer_aggregate_load_helpers()
        code += self.generate_metal_inverse_helpers()
        code += functions_code
        return code

    def collect_metal_atomic_fence_calls(self, ast):
        if "atomicThreadFence" in self.user_function_names:
            return []
        walk = getattr(ast, "walk", None)
        nodes = walk() if callable(walk) else self.iter_ast_nodes(ast)
        return [
            node
            for node in nodes
            if isinstance(node, FunctionCallNode)
            and self.function_call_name(node) == "atomicThreadFence"
        ]

    def collect_metal_resource_memory_contracts(self, ast):
        walk = getattr(ast, "walk", None)
        nodes = walk() if callable(walk) else self.iter_ast_nodes(ast)
        contracts = []
        for node in nodes:
            raw_type = getattr(node, "param_type", getattr(node, "var_type", None))
            if not self.metal_resource_memory_contract_is_emittable(node, raw_type):
                continue
            for contract in self.resource_memory_qualifier_contracts(node, raw_type):
                if contract not in contracts:
                    contracts.append(contract)
        return contracts

    def metal_resource_memory_contract_is_emittable(self, node, raw_type):
        contract_type = raw_type
        while self.is_array_type_node(contract_type):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            return True
        if self.is_structured_buffer_type(raw_type):
            return True

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        qualifiers.update(
            str(qualifier).lower()
            for qualifier in getattr(node, "resource_qualifiers", []) or []
        )
        qualifiers.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        )
        return bool(
            qualifiers
            & {
                "constant",
                "device",
                "global",
                "local",
                "private",
                "storage",
                "thread",
                "threadgroup",
                "workgroup",
            }
        )

    @staticmethod
    def generate_metal_system_thread_scope_support():
        return (
            "#ifndef __METAL_MEMORY_SCOPE_SYSTEM__\n"
            "#define __METAL_MEMORY_SCOPE_SYSTEM__ 3\n"
            "#endif\n"
            "namespace metal {\n"
            "constexpr constant thread_scope thread_scope_system =\n"
            "    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);\n"
            "}\n\n"
        )

    def uses_cooperative_matrix(self, ast):
        """Return whether a canonical cooperative-matrix node is present."""
        walk = getattr(ast, "walk", None)
        if not callable(walk):
            return False
        return any(
            isinstance(node, (CooperativeMatrixType, CooperativeMatrixOpNode))
            for node in walk()
        )

    def generate_preprocessor_directive(self, directive):
        if isinstance(directive, PreprocessorNode):
            directive_name = (directive.directive or "").strip()
            if directive_name.lower() in {"version", "extension", "precision"}:
                return None
            return f"#{directive_name} {directive.content}".strip()

        line = str(directive).strip()
        lowered = line.lower()
        if (
            lowered.startswith("#version")
            or lowered.startswith("#extension")
            or lowered.startswith("precision ")
        ):
            return None
        return line

    def generate_constants(self, ast, constants=None, include_function_constants=True):
        code = ""
        for node in (
            getattr(ast, "constants", []) or [] if constants is None else constants
        ):
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = self.generate_constant_expression(value)
            mapped_type = self.map_type(const_type)
            specialization_id = self.metal_specialization_constant_id(node)
            if specialization_id is not None:
                code += (
                    "/* CrossGL fallback: Metal source output cannot preserve "
                    f"GLSL specialization constant id {specialization_id} for "
                    f"'{name}'; using the default literal. */\n"
                )
            if specialization_id is not None and mapped_type == "double":
                code += (
                    "/* CrossGL fallback: Metal does not support double "
                    f"specialization constant '{name}'; lowered to float. */\n"
                )
                mapped_type = "float"
            declaration = format_c_style_array_declaration(mapped_type, name)
            code += (
                f"{self.metal_unused_declaration_qualifier('constant')} "
                f"{declaration} = {value_code};\n"
            )

        if not include_function_constants:
            return f"{code}\n" if code else ""

        used_function_constant_ids = {}
        for node in getattr(ast, "global_variables", []) or []:
            function_constant_id = self.metal_function_constant_id(node)
            if function_constant_id is None:
                continue

            name = getattr(node, "name", getattr(node, "variable_name", None))
            previous_name = used_function_constant_ids.get(function_constant_id)
            if previous_name is not None:
                raise ValueError(
                    "Duplicate Metal function constant id "
                    f"{function_constant_id} for '{name}' and '{previous_name}'"
                )
            used_function_constant_ids[function_constant_id] = name

            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            self.validate_metal_function_constant_type(node, var_type)
            mapped_type = self.map_type(var_type)
            initial_value = getattr(node, "initial_value", None)
            if initial_value is not None:
                code += (
                    "/* unsupported Metal function constant default: "
                    f"'{name}' initializers are not allowed by MSL */\n"
                )
            code += (
                f"{self.metal_unused_declaration_qualifier('constant')} "
                f"{mapped_type} {name} "
                f"[[function_constant({function_constant_id})]];\n"
            )

        return f"{code}\n" if code else ""

    def metal_unused_declaration_qualifier(self, qualifier):
        return f"__attribute__((unused)) {qualifier}"

    def format_unused_metal_declaration(self, declaration):
        declaration = str(declaration).strip()
        if declaration.startswith("__attribute__((unused)) "):
            return declaration
        return f"__attribute__((unused)) {declaration}"

    def maybe_format_unused_local_declaration(self, declaration, name):
        if name in self.current_unused_local_declaration_names:
            return self.format_unused_metal_declaration(declaration)
        return declaration

    def strip_unused_metal_declaration_attribute(self, declaration):
        declaration = str(declaration).strip()
        prefix = "__attribute__((unused)) "
        if declaration.startswith(prefix):
            return declaration[len(prefix) :]
        return declaration

    def format_match_binding_declaration(self, declaration, _binding_name):
        return self.format_unused_metal_declaration(declaration)

    def metal_function_constant_attributes(self, node):
        attributes = []
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower()
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized in {"function_constant", "constant_id"}:
                attributes.append(attr)
        return attributes

    def metal_specialization_constant_id(self, node):
        constant_id = None
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower().replace("-", "_")
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized != "constant_id":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            value = self.attribute_value_to_string(arguments[0])
            if constant_id is not None and constant_id != value:
                name = getattr(
                    node, "name", getattr(node, "variable_name", "<unnamed>")
                )
                raise ValueError(
                    "Conflicting Metal specialization constant ids for "
                    f"'{name}': {constant_id} differs from {value}"
                )
            constant_id = value
        return constant_id

    def is_metal_function_constant_variable(self, node):
        return bool(self.metal_function_constant_attributes(node))

    def normalized_metal_abi_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None
        normalized = str(attr_name).lower()
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return normalized

    def metal_function_constant_id(self, node):
        attributes = self.metal_function_constant_attributes(node)
        if not attributes:
            return None

        name = getattr(node, "name", getattr(node, "variable_name", None))
        if len(attributes) > 1:
            raise ValueError(
                f"Metal function constant '{name}' has multiple function constant "
                "attributes"
            )

        arguments = getattr(attributes[0], "arguments", []) or []
        function_constant_id = (
            self.binding_index_value(arguments[0]) if len(arguments) == 1 else None
        )
        if function_constant_id is None:
            raise ValueError(f"Metal function constant '{name}' requires an integer id")
        return function_constant_id

    def validate_metal_function_constant_type(self, node, var_type):
        name = getattr(node, "name", getattr(node, "variable_name", None))
        type_name = self.type_name_string(var_type)
        mapped_type = self.map_type(var_type)
        invalid = (
            self.is_resource_parameter_type(type_name)
            or self.is_array_type_node(var_type)
            or "[" in mapped_type
            or "]" in mapped_type
            or "*" in mapped_type
            or "&" in mapped_type
        )
        if invalid:
            raise ValueError(
                f"Metal function constant '{name}' cannot use type '{type_name}'"
            )

        if not self.is_metal_function_constant_value_type(mapped_type):
            raise ValueError(
                f"Metal function constant '{name}' cannot use type '{type_name}'"
            )

    def is_metal_function_constant_value_type(self, mapped_type):
        scalar_types = {
            "bool",
            "int",
            "uint",
            "int64_t",
            "uint64_t",
            "float",
            "half",
        }
        if mapped_type in scalar_types:
            return True

        for base_type in scalar_types:
            if mapped_type in {
                f"{base_type}2",
                f"{base_type}3",
                f"{base_type}4",
            }:
                return True
        return False

    def is_metal_struct_member_abi_attribute(self, attr):
        return self.normalized_metal_abi_attribute_name(attr) == "id"

    def is_metal_argument_buffer_global(self, node):
        attributes = getattr(node, "attributes", []) or []
        return any(
            self.normalized_metal_abi_attribute_name(attr) == "argument_buffer"
            for attr in attributes
        )

    def format_metal_argument_buffer_global(self, node, mapped_type, name):
        if not name:
            return None

        attributes = []
        seen_attributes = set()
        for attr in getattr(node, "attributes", []) or []:
            attr_name = self.normalized_metal_abi_attribute_name(attr)
            if attr_name not in {"buffer", "id", "argument_buffer"}:
                continue
            if attr_name in seen_attributes:
                raise ValueError(
                    f"Metal argument buffer global '{name}' has multiple "
                    f"{attr_name} attributes"
                )
            seen_attributes.add(attr_name)

            arguments = getattr(attr, "arguments", []) or []
            if attr_name in {"buffer", "id"}:
                attr_value = (
                    self.binding_index_value(arguments[0])
                    if len(arguments) == 1
                    else None
                )
                if attr_value is None:
                    raise ValueError(
                        f"Metal argument buffer global '{name}' requires an "
                        f"integer {attr_name} attribute"
                    )
                attributes.append(f"[[{attr_name}({attr_value})]]")
            else:
                attributes.append("[[argument_buffer]]")

        if "buffer" not in seen_attributes:
            raise ValueError(
                f"Metal argument buffer global '{name}' requires a buffer attribute"
            )

        base_type = str(mapped_type).strip().rstrip("&").strip()
        abi_attributes = " ".join(attributes)
        qualifier = self.metal_unused_declaration_qualifier("constant")
        return f"{qualifier} {base_type}& {name} {abi_attributes};\n"

    def metal_argument_buffer_binding(self, node):
        if not self.is_metal_argument_buffer_global(node):
            return None
        return self.explicit_resource_binding_index(node, {"buffer"}, ("b",))

    def format_metal_struct_member_abi_attributes(self, member):
        attributes = []
        seen_attributes = set()
        member_name = getattr(member, "name", None)
        for attr in getattr(member, "attributes", []) or []:
            if not self.is_metal_struct_member_abi_attribute(attr):
                continue

            arguments = getattr(attr, "arguments", []) or []
            attr_name = self.normalized_metal_abi_attribute_name(attr)

            if attr_name in seen_attributes:
                raise ValueError(
                    f"Metal struct member '{member_name}' has multiple {attr_name} "
                    "attributes"
                )
            seen_attributes.add(attr_name)

            if attr_name == "id":
                member_id = (
                    self.binding_index_value(arguments[0])
                    if len(arguments) == 1
                    else None
                )
                if member_id is None:
                    raise ValueError(
                        f"Metal struct member '{member_name}' requires an integer id"
                    )
                attributes.append(f"[[id({member_id})]]")

        return f" {' '.join(attributes)}" if attributes else ""

    def metal_interpolation_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None
        normalized = str(attr_name).lower()
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return self.METAL_INTERPOLATION_ATTRIBUTES.get(normalized)

    def metal_interpolation_attribute_suffix(self, node):
        for attr in getattr(node, "attributes", []) or []:
            interpolation = self.metal_interpolation_attribute_name(attr)
            if interpolation is not None:
                return f" [[{interpolation}]]"
        implied = self.metal_semantic_implied_interpolation_attribute(node)
        if implied is not None:
            return f" [[{implied}]]"
        return ""

    def metal_semantic_implied_interpolation_attribute(self, node):
        semantic = self.semantic_from_node(node)
        if semantic is None:
            return None
        if str(semantic).lower() == "gl_barycoordnoperspext":
            return "center_no_perspective"
        return None

    def format_struct_resource_array_member(self, member):
        name = getattr(member, "name", None)
        if not name:
            return None

        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", None)
            )
            if element_type is None:
                return None
            base_type = (
                self.convert_type_node_to_string(element_type)
                if hasattr(element_type, "name")
                or hasattr(element_type, "element_type")
                else str(element_type)
            )
            if not self.is_resource_parameter_type(base_type):
                return None
            resource_type = self.map_resource_type_with_format(base_type, member)
            if member.size:
                array_size = self.safe_expression_to_string(member.size)
                return f"array<{resource_type}, {array_size}> {name}"
            return f"array<{resource_type}> {name}"

        raw_member_type = getattr(member, "member_type", None)
        if raw_member_type is None or not self.is_array_type_node(raw_member_type):
            return None

        array_type = self.resource_array_parameter(raw_member_type, member)
        if array_type is None:
            return None
        resource_type, array_size = array_type
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{resource_type}, {array_size}> {name}"
        return f"array<{resource_type}> {name}"

    def generate_constant_expression(self, expr):
        value_code = self.generate_expression(expr)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def synthetic_hlsl_program_constants_cbuffer(self, ast):
        members = []
        self.hlsl_program_constant_global_ids = set()
        for node in getattr(ast, "global_variables", []) or []:
            if not self.is_hlsl_program_constant_global(node):
                continue
            members.append(node)
            self.hlsl_program_constant_global_ids.add(id(node))

        if not members:
            return None

        cbuffer = StructNode(
            self.unique_hlsl_program_constants_cbuffer_name(ast),
            members,
        )
        cbuffer.is_cbuffer = True
        cbuffer.buffer_kind = "cbuffer"
        cbuffer.is_synthetic_hlsl_program_constants = True
        return cbuffer

    def is_hlsl_program_constant_global(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() == "hlsl_program_constant":
                return True
        return False

    def unique_hlsl_program_constants_cbuffer_name(self, ast):
        reserved_names = {
            getattr(node, "name", None)
            for node in (
                list(getattr(ast, "structs", []) or [])
                + list(getattr(ast, "cbuffers", []) or [])
                + list(getattr(ast, "global_variables", []) or [])
            )
            if getattr(node, "name", None)
        }
        return self.unique_metal_generated_name("HlslProgramConstants", reserved_names)

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = self.cbuffer_variables or getattr(ast, "cbuffers", [])
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate cbuffer name(s) in Metal output: {names}")

        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(ast)
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise ValueError(
                f"Cbuffer name(s) conflict with existing Metal declaration(s): {names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(ast)
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise ValueError(
                "Cbuffer member name(s) conflict with Metal global declaration(s): "
                f"{names}"
            )

        for node in cbuffers:
            if isinstance(node, StructNode):
                code += f"struct {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Metal requires a concrete element count for value arrays.
                            code += (
                                f"    {self.map_type(element_type)} "
                                f"{member.name}[1024];\n"
                            )
                    else:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # CbufferNode handling
                code += f"struct {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Metal requires a concrete element count for value arrays.
                            code += (
                                f"    {self.map_type(element_type)} "
                                f"{member.name}[1024];\n"
                            )
                    else:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"

        return code

    def generate_metal_struct_declarations(self, structs):
        definitions = []

        for specialization in (self.generic_struct_specializations or {}).values():
            fields = generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            )
            definitions.append(
                self.metal_field_struct_definition(
                    specialization["struct_name"],
                    fields,
                )
            )

        for enum in self.struct_payload_enums or []:
            fields = [("variant", "int")]
            fields.extend(enum_struct_fields(enum) or [])
            definitions.append(self.metal_field_struct_definition(enum.name, fields))

        for specialization in (self.generic_enum_specializations or {}).values():
            fields = [("variant", "int")]
            fields.extend(generic_enum_specialized_fields(self, specialization))
            definitions.append(
                self.metal_field_struct_definition(
                    specialization["struct_name"],
                    fields,
                )
            )

        for node in structs or []:
            definition = self.metal_plain_struct_definition(node)
            if definition is not None:
                definitions.append(definition)

        self.metal_generated_struct_names = {
            definition["name"] for definition in definitions if definition.get("name")
        }
        return "".join(
            definition["code"]
            for definition in self.order_metal_struct_definitions(definitions)
        )

    def metal_field_struct_definition(self, name, fields):
        code = f"struct {name} {{\n"
        dependencies = set()
        for field_name, field_type in fields:
            mapped_type = self.map_type(field_type)
            declaration = format_c_style_array_declaration(mapped_type, field_name)
            code += f"    {declaration};\n"
            dependencies.update(self.metal_struct_type_dependencies(mapped_type))
        code += "};\n\n"
        dependencies.discard(name)
        return {"name": name, "dependencies": dependencies, "code": code}

    def metal_plain_struct_definition(self, node):
        if not isinstance(node, StructNode):
            return None
        if node.name in self.generic_enum_struct_definitions:
            return None
        if node.name in self.generic_struct_definitions:
            return None
        if node.name in self.lowered_glsl_buffer_block_struct_names:
            return None
        if node.name in self.glsl_buffer_block_struct_names:
            code = self.glsl_buffer_block_diagnostic("Metal", node.name, None, None)
            code += self.unsupported_glsl_buffer_block_struct_placeholder(
                "Metal", node.name
            )
            return {"name": node.name, "dependencies": set(), "code": code}

        self.validate_struct_member_semantic_types(node)
        code = f"struct {node.name} {{\n"
        dependencies = set()
        default_member_semantics = self.metal_default_struct_member_semantics(node)
        for member in getattr(node, "members", []) or []:
            member_code, member_dependencies = self.metal_struct_member_declaration(
                member,
                default_member_semantics,
                node.name,
            )
            code += member_code
            dependencies.update(member_dependencies)
        fallback_position = self.metal_vertex_output_position_fallback_member_name(node)
        if fallback_position is not None:
            code += (
                "    /* CrossGL fallback: Metal vertex entry points require "
                "a position output even when the GLSL source did not write "
                "gl_Position. */\n"
                f"    float4 {fallback_position} [[position]];\n"
            )
        code += "};\n"
        dependencies.discard(node.name)
        return {"name": node.name, "dependencies": dependencies, "code": code}

    def collect_metal_stage_io_member_lowerings(self, structs):
        lowerings = {}
        stage_io_names = set(getattr(self, "metal_stage_io_struct_names", set()))
        if not stage_io_names:
            return lowerings

        for struct in structs or []:
            struct_name = getattr(struct, "name", None)
            if struct_name not in stage_io_names:
                continue
            default_member_semantics = self.metal_default_struct_member_semantics(
                struct
            )
            for member in getattr(struct, "members", []) or []:
                lowering = self.metal_stage_io_member_lowering(
                    member, default_member_semantics, struct_name
                )
                if lowering is not None:
                    lowerings.setdefault(struct_name, {})[member.name] = lowering
        return lowerings

    def metal_stage_io_member_lowering(
        self, member, default_member_semantics=None, struct_name=None
    ):
        member_name = getattr(member, "name", None)
        if not member_name:
            return None
        semantic = self.metal_struct_member_effective_semantic(
            member, default_member_semantics or {}, struct_name
        )

        array_info = self.metal_stage_io_member_array_info(member)
        if array_info is not None:
            element_type, size = array_info
            fields = [
                (self.metal_stage_io_array_field_name(member_name, index), element_type)
                for index in range(size)
            ]
            return {
                "kind": "array",
                "element_type": element_type,
                "size": size,
                "fields": fields,
                "field_semantics": self.metal_stage_io_lowered_field_semantics(
                    semantic, len(fields)
                ),
                "interpolation_attr": self.metal_interpolation_attribute_suffix(member),
            }

        raw_type = self.struct_member_raw_type(member)
        mapped_type = self.map_type(raw_type)
        dimensions = self.metal_matrix_dimensions(mapped_type)
        if dimensions is None:
            return None
        component_prefix, columns, rows = dimensions
        column_type = f"{component_prefix}{rows}"
        fields = [
            (self.metal_stage_io_matrix_field_name(member_name, index), column_type)
            for index in range(columns)
        ]
        return {
            "kind": "matrix",
            "mapped_type": mapped_type,
            "columns": columns,
            "rows": rows,
            "fields": fields,
            "field_semantics": self.metal_stage_io_lowered_field_semantics(
                semantic, len(fields)
            ),
            "interpolation_attr": self.metal_interpolation_attribute_suffix(member),
        }

    def metal_stage_io_lowered_field_semantics(self, semantic, field_count):
        attribute_index = self.metal_attribute_index_from_semantic(semantic)
        if attribute_index is None:
            return [None] * field_count
        return [f"attribute({attribute_index + index})" for index in range(field_count)]

    def metal_stage_io_member_array_info(self, member):
        if self.canonical_metal_semantic(self.semantic_from_node(member)) in {
            "clip_distance"
        }:
            return None

        if isinstance(member, ArrayNode):
            resource_array_declaration = self.format_struct_resource_array_member(
                member
            )
            if resource_array_declaration is not None:
                return None
            element_type = self.map_type(
                getattr(member, "element_type", getattr(member, "vtype", "float"))
            )
            size = self.literal_int_value(member.size, self.literal_int_constants)
            return (element_type, size) if size else None

        raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if self.is_array_type_node(raw_type):
            element_type = self.map_type(raw_type.element_type)
            size = self.literal_int_value(raw_type.size, self.literal_int_constants)
            return (element_type, size) if size else None

        raw_type_name = self.type_name_string(raw_type)
        if raw_type_name and "[" in raw_type_name and "]" in raw_type_name:
            element_type, array_size = parse_array_type(raw_type_name)
            size = self.literal_int_value(array_size, self.literal_int_constants)
            return (self.map_type(element_type), size) if size else None
        return None

    def metal_matrix_dimensions(self, mapped_type):
        mapped_type = self.map_type(mapped_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return None
        for component_prefix in ("float", "half", "double"):
            if not base_type.startswith(component_prefix):
                continue
            suffix = base_type[len(component_prefix) :]
            if "x" not in suffix:
                return None
            columns, rows = suffix.split("x", 1)
            if columns in {"2", "3", "4"} and rows in {"2", "3", "4"}:
                return component_prefix, int(columns), int(rows)
        return None

    def generate_metal_matrix_resize_constructor(self, target_type, args):
        target_dimensions = self.metal_matrix_dimensions(target_type)
        if target_dimensions is None or len(args or []) != 1:
            return None

        arg_type = self.expression_result_type(args[0])
        arg_dimensions = self.metal_matrix_dimensions(arg_type)
        if arg_dimensions is None:
            return None

        target_prefix, target_columns, target_rows = target_dimensions
        _arg_prefix, arg_columns, arg_rows = arg_dimensions
        if target_columns == arg_columns and target_rows == arg_rows:
            return None

        rendered_arg = self.generate_expression_with_expected(args[0], None)
        columns = []
        for column in range(target_columns):
            if column < arg_columns:
                source_column = f"{rendered_arg}[{column}]"
                if target_rows == arg_rows:
                    columns.append(source_column)
                elif target_rows < arg_rows:
                    columns.append(
                        f"{source_column}.{self.vector_swizzle(target_rows)}"
                    )
                else:
                    fill_values = [
                        "1.0" if row == column else "0.0"
                        for row in range(arg_rows, target_rows)
                    ]
                    columns.append(
                        f"{target_prefix}{target_rows}({source_column}, {', '.join(fill_values)})"
                    )
            else:
                values = [
                    "1.0" if row == column else "0.0" for row in range(target_rows)
                ]
                columns.append(f"{target_prefix}{target_rows}({', '.join(values)})")
        return f"{target_type}({', '.join(columns)})"

    def vector_swizzle(self, width):
        return {1: "x", 2: "xy", 3: "xyz", 4: "xyzw"}[width]

    def metal_stage_io_matrix_field_name(self, member_name, index):
        return f"{member_name}_{index}"

    def metal_stage_io_array_field_name(self, member_name, index):
        return f"{member_name}_{index}"

    def generate_metal_stage_io_array_helpers(self):
        code = ""
        for struct_name in sorted(self.metal_stage_io_member_lowerings):
            for member_name, lowering in sorted(
                self.metal_stage_io_member_lowerings[struct_name].items()
            ):
                if lowering.get("kind") != "array":
                    continue
                helper_name = self.metal_stage_io_array_getter_name(
                    struct_name, member_name
                )
                element_type = lowering["element_type"]
                code += (
                    f"static inline {element_type} __attribute__((unused)) {helper_name}"
                    f"({struct_name} value, int index) {{\n"
                    "    switch (index) {\n"
                )
                for index, (field_name, _field_type) in enumerate(lowering["fields"]):
                    code += f"    case {index}: return value.{field_name};\n"
                code += (
                    "    default: "
                    f"return {self.metal_default_value_expression(element_type)};\n"
                    "    }\n"
                    "}\n\n"
                )
        return code

    def metal_stage_io_array_getter_name(self, struct_name, member_name):
        return f"__crossgl_stage_io_get_{struct_name}_{member_name}"

    def metal_struct_member_declaration(
        self, member, default_member_semantics, struct_name=None
    ):
        dependencies = set()
        lowering = self.metal_stage_io_member_lowerings.get(struct_name, {}).get(
            getattr(member, "name", None)
        )
        if lowering is not None:
            code = ""
            field_semantics = lowering.get("field_semantics", [])
            interpolation_attr = lowering.get("interpolation_attr", "")
            for index, (field_name, field_type) in enumerate(lowering["fields"]):
                semantic = (
                    field_semantics[index] if index < len(field_semantics) else None
                )
                semantic_attr = self.map_semantic(semantic) if semantic else ""
                code += (
                    f"    {field_type} {field_name}{semantic_attr}"
                    f"{interpolation_attr};\n"
                )
                dependencies.update(self.metal_struct_type_dependencies(field_type))
            return code, dependencies

        if isinstance(member, ArrayNode):
            semantic = self.metal_struct_member_effective_semantic(
                member, default_member_semantics, struct_name
            )
            resource_array_declaration = self.format_struct_resource_array_member(
                member
            )
            if resource_array_declaration is not None:
                abi_attr = self.format_metal_struct_member_abi_attributes(member)
                semantic_attr = self.map_semantic(semantic) if semantic else ""
                interpolation_attr = self.metal_interpolation_attribute_suffix(member)
                return (
                    f"    {resource_array_declaration}{abi_attr}{semantic_attr}"
                    f"{interpolation_attr};\n",
                    set(),
                )

            element_type = getattr(
                member,
                "element_type",
                getattr(member, "vtype", "float"),
            )
            semantic_attr = self.map_semantic(semantic) if semantic else ""
            interpolation_attr = self.metal_interpolation_attribute_suffix(member)
            mapped_type = self.map_type(element_type)
            dependencies.update(self.metal_struct_type_dependencies(mapped_type))
            if member.size:
                if self.metal_array_semantic_attribute_precedes_extent(semantic):
                    return (
                        f"    {mapped_type} {member.name}{semantic_attr}"
                        f"{interpolation_attr}[{member.size}];\n",
                        dependencies,
                    )
                return (
                    f"    {mapped_type} {member.name}[{member.size}]"
                    f"{semantic_attr}{interpolation_attr};\n",
                    dependencies,
                )
            if self.metal_array_semantic_attribute_precedes_extent(semantic):
                return (
                    f"    {mapped_type} {member.name}{semantic_attr}"
                    f"{interpolation_attr}[1024];\n",
                    dependencies,
                )
            return (
                f"    {mapped_type} {member.name}[1024]{semantic_attr}"
                f"{interpolation_attr};\n",
                dependencies,
            )

        semantic = self.metal_struct_member_effective_semantic(
            member, default_member_semantics, struct_name
        )
        abi_attr = self.format_metal_struct_member_abi_attributes(member)
        semantic_attr = self.map_semantic(semantic) if semantic else ""
        interpolation_attr = self.metal_interpolation_attribute_suffix(member)

        if hasattr(member, "member_type"):
            if self.is_array_type_node(member.member_type):
                resource_array_declaration = self.format_struct_resource_array_member(
                    member
                )
                if resource_array_declaration is not None:
                    return (
                        f"    {resource_array_declaration}{abi_attr}{semantic_attr}"
                        f"{interpolation_attr};\n",
                        set(),
                    )
            address_space_declaration = self.format_address_space_parameter_declaration(
                member.member_type,
                self.map_type(self.convert_type_node_to_string(member.member_type)),
                member.name,
                member,
            )
            if address_space_declaration is not None:
                dependencies.update(
                    self.metal_struct_type_dependencies(
                        self.convert_type_node_to_string(member.member_type)
                    )
                )
                return (
                    f"    {address_space_declaration}{abi_attr}{semantic_attr}"
                    f"{interpolation_attr};\n",
                    dependencies,
                )
            if str(type(member.member_type)).find("ArrayType") != -1:
                member_type_str = self.convert_type_node_to_string(member.member_type)
                member_type = self.map_type(member_type_str)
                dependencies.update(self.metal_struct_type_dependencies(member_type))
                declaration = format_c_style_array_declaration(member_type, member.name)
                if self.metal_array_semantic_attribute_precedes_extent(semantic):
                    base_type, array_size = split_array_type_suffix(member_type)
                    if member.member_type.size is None:
                        array_size = "1024"
                    if not array_size:
                        array_size = self.safe_expression_to_string(
                            member.member_type.size
                        )
                    array_size = str(array_size).strip()
                    if array_size.startswith("[") and array_size.endswith("]"):
                        array_size = array_size[1:-1]
                    return (
                        f"    {base_type} {member.name}{abi_attr}{semantic_attr}"
                        f"{interpolation_attr}[{array_size}];\n",
                        dependencies,
                    )
                if member.member_type.size is None:
                    base_type, _ = split_array_type_suffix(member_type)
                    return (
                        f"    {base_type} {member.name}[1024]{abi_attr}"
                        f"{semantic_attr}{interpolation_attr};\n",
                        dependencies,
                    )
                return (
                    f"    {declaration}{abi_attr}{semantic_attr}"
                    f"{interpolation_attr};\n",
                    dependencies,
                )

            member_type_str = self.convert_type_node_to_string(member.member_type)
            member_type = self.map_resource_type_with_format(member_type_str, member)
        elif hasattr(member, "vtype"):
            member_type = self.map_resource_type_with_format(member.vtype, member)
        else:
            member_type = "float"

        dependencies.update(self.metal_struct_type_dependencies(member_type))
        return (
            f"    {member_type} {member.name}{abi_attr}{semantic_attr}"
            f"{interpolation_attr};\n",
            dependencies,
        )

    def metal_struct_type_dependencies(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return set()
        mapped_type = self.map_type(type_name)
        base_type, _ = split_array_type_suffix(str(mapped_type))
        base_type = base_type.strip()
        if base_type.startswith("array<"):
            base_type = base_type[len("array<") :].split(",", 1)[0].rstrip(">")
        base_type = base_type.replace("const ", "").strip()
        for separator in ("&", "*"):
            if separator in base_type:
                base_type = base_type.split(separator, 1)[0].strip()
        if "<" in base_type:
            base_type = base_type.split("<", 1)[0].strip()
        return {base_type}

    def order_metal_struct_definitions(self, definitions):
        generated_names = {
            definition["name"] for definition in definitions if definition.get("name")
        }
        remaining = [
            {
                **definition,
                "dependencies": (
                    set(definition.get("dependencies", set())) & generated_names
                ),
            }
            for definition in definitions
        ]
        ordered = []
        emitted = set()

        while remaining:
            progressed = False
            for definition in list(remaining):
                if definition["dependencies"] <= emitted:
                    ordered.append(definition)
                    emitted.add(definition["name"])
                    remaining.remove(definition)
                    progressed = True
            if not progressed:
                ordered.extend(remaining)
                break

        return ordered

    def add_metal_generic_specializations_from_type_text(self, type_text, visited=None):
        type_text = normalize_specialization_type_text(type_text)
        if not type_text:
            return
        visited = set(visited or set())
        if type_text in visited:
            return
        visited.add(type_text)

        base_name, generic_args = generic_type_parts(type_text)
        for arg in generic_args:
            self.add_metal_generic_specializations_from_type_text(arg, visited)

        enum_definition = self.generic_enum_struct_definitions.get(base_name)
        if enum_definition is not None and len(generic_args) == len(
            enum_definition["generic_params"]
        ):
            specialization = self.generic_enum_specializations.get(type_text)
            if specialization is None:
                specialization = build_generic_enum_specialization(
                    type_text,
                    enum_definition,
                )
                self.generic_enum_specializations[type_text] = specialization
            for _field_name, field_type in generic_enum_specialized_fields(
                self,
                specialization,
            ):
                self.add_metal_generic_specializations_from_type_text(
                    field_type, visited
                )
            return

        struct_definition = self.generic_struct_definitions.get(base_name)
        if struct_definition is not None and len(generic_args) == len(
            struct_definition["generic_params"]
        ):
            specialization = self.generic_struct_specializations.get(type_text)
            if specialization is None:
                specialization = build_generic_struct_specialization(
                    type_text,
                    struct_definition,
                )
                self.generic_struct_specializations[type_text] = specialization
            for _field_name, field_type in generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            ):
                self.add_metal_generic_specializations_from_type_text(
                    field_type, visited
                )

    def generate_metal_enum_constructor_functions(self, enums):
        code = ""
        for enum in enums or []:
            all_fields = enum_struct_fields(enum) or []
            for variant in enum.variants or []:
                variant_fields = enum_variant_payload_fields(variant) or []
                params = ", ".join(
                    f"{self.map_type(field_type)} payload{index}"
                    for index, (_field_name, field_type) in enumerate(variant_fields)
                )
                code += (
                    f"{enum.name} "
                    f"{enum_variant_constructor_name(enum.name, variant.name)}"
                    f"({params}) {{\n"
                )
                code += f"    {enum.name} result;\n"
                code += (
                    f"    result.variant = "
                    f"{enum_variant_constructor_name(enum.name, variant.name).rsplit('_make', 1)[0]};\n"
                )
                active_fields = {
                    field_name for field_name, _field_type in variant_fields
                }
                for field_name, field_type in all_fields:
                    if field_name in active_fields:
                        continue
                    code += (
                        f"    result.{field_name} = "
                        f"{self.metal_default_value_expression(field_type)};\n"
                    )
                for index, (field_name, _field_type) in enumerate(variant_fields):
                    code += f"    result.{field_name} = payload{index};\n"
                code += "    return result;\n"
                code += "}\n\n"
        return code

    def generate_metal_generic_enum_constructor_functions(self, specializations):
        code = ""
        for specialization in (specializations or {}).values():
            all_fields = generic_enum_specialized_fields(self, specialization)
            definition = specialization["definition"]
            for variant in definition["enum"].variants or []:
                variant_fields = (
                    generic_enum_specialized_variant_fields(
                        self,
                        specialization,
                        variant.name,
                    )
                    or []
                )
                params = ", ".join(
                    f"{self.map_type(field_type)} payload{index}"
                    for index, (_field_name, field_type) in enumerate(variant_fields)
                )
                code += (
                    f"{specialization['struct_name']} "
                    f"{enum_variant_constructor_name(specialization['struct_name'], variant.name)}"
                    f"({params}) {{\n"
                )
                code += f"    {specialization['struct_name']} result;\n"
                code += (
                    "    result.variant = "
                    f"{enum_variant_constructor_name(definition['name'], variant.name).rsplit('_make', 1)[0]};\n"
                )
                active_fields = {
                    field_name for field_name, _field_type in variant_fields
                }
                for field_name, field_type in all_fields:
                    if field_name in active_fields:
                        continue
                    code += (
                        f"    result.{field_name} = "
                        f"{self.metal_default_value_expression(field_type)};\n"
                    )
                for index, (field_name, _field_type) in enumerate(variant_fields):
                    code += f"    result.{field_name} = payload{index};\n"
                code += "    return result;\n"
                code += "}\n\n"
        return code

    def metal_default_value_expression(self, field_type):
        mapped_type = self.map_type(field_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return f"{base_type}()"
        if (
            base_type in self.metal_generated_struct_names
            or base_type in self.struct_member_types
            or base_type in self.structs_by_name
        ):
            return f"{base_type}()"
        if base_type == "bool":
            return "false"
        if base_type.startswith("bool") or base_type.startswith("bvec"):
            return f"{base_type}(false)"
        return f"{base_type}(0)"

    def is_input_attachment_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        return bool(re.fullmatch(r"[iu]?subpassInput(?:MS)?", str(type_name or "")))

    def unsupported_input_attachment_call(self, func_name):
        if func_name != "subpassLoad":
            return None
        fallback_type = self.current_expression_expected_type or "vec4"
        fallback = self.metal_default_value_expression(fallback_type)
        return (
            "/* unsupported Metal input attachment load: "
            "subpassLoad requires Vulkan subpass input lowering */ "
            f"{fallback}"
        )

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        stage_local_variables=None,
        stage_node=None,
    ):
        """Render a function or stage entry point with Metal attributes."""
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        reserved_parameter_names = {
            getattr(parameter, "name", None)
            for parameter in param_list
            if getattr(parameter, "name", None)
        }
        sampler_parameters = set()
        sampler_parameter_array_sizes = {}
        texture_parameters = {}
        texture_parameter_raw_types = {}
        texture_parameter_array_sizes = {}
        image_format_parameters = {}
        vertex_stage_input_parameters = []
        fragment_stage_input_parameters = []
        stage_builtin_parameter_alias_declarations = []
        vertex_stage_input_alias_declarations = []
        fragment_stage_input_alias_declarations = []
        stage_output_parameters = []
        stage_output_struct_name = None
        previous_function_name = self.current_function_name
        previous_function_return_type = self.current_function_return_type
        previous_function_return_wrapper = self.current_function_return_wrapper
        previous_local_identifier_remaps = self.current_local_identifier_remaps
        previous_stage_output_return_active = getattr(
            self, "current_metal_stage_output_return_active", False
        )
        previous_local_variable_types = self.local_variable_types
        previous_address_space_variables = self.current_address_space_variables
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        previous_cbuffer_parameter_names = self.cbuffer_parameter_names
        previous_cbuffer_member_references = self.cbuffer_member_references
        previous_ambiguous_cbuffer_members = self.ambiguous_cbuffer_members
        previous_glsl_buffer_block_parameters = (
            self.current_glsl_buffer_block_parameters
        )
        previous_unsupported_glsl_buffer_block_parameters = (
            self.current_unsupported_glsl_buffer_block_parameters
        )
        previous_unsupported_glsl_buffer_block_local_variables = (
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        previous_glsl_buffer_block_parameter_failures = (
            self.current_glsl_buffer_block_parameter_failures
        )
        previous_glsl_buffer_block_parameter_struct_failures = (
            self.current_glsl_buffer_block_parameter_struct_failures
        )
        previous_unsupported_metal_ray_function_table_array_variables = dict(
            self.unsupported_metal_ray_function_table_array_variables
        )
        previous_unsupported_metal_acceleration_structure_array_variables = dict(
            self.unsupported_metal_acceleration_structure_array_variables
        )
        previous_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        previous_structured_buffer_counter_parameters = (
            self.current_structured_buffer_counter_parameters
        )
        previous_metal_mesh_output_config = self.current_metal_mesh_output_config
        previous_metal_mesh_output_parameter = self.current_metal_mesh_output_parameter
        previous_metal_mesh_grid_properties_parameter = (
            self.current_metal_mesh_grid_properties_parameter
        )
        previous_metal_mesh_payload_parameter = (
            self.current_metal_mesh_payload_parameter
        )
        previous_metal_mesh_payload_type = self.current_metal_mesh_payload_type
        previous_metal_mesh_output_accumulators = (
            self.current_metal_mesh_output_accumulators
        )
        previous_metal_non_thread_payload_parameters = (
            self.current_metal_non_thread_payload_parameters
        )
        previous_readonly_metal_mesh_payload_parameters = (
            self.current_readonly_metal_mesh_payload_parameters
        )
        previous_readonly_metal_mesh_payload_reasons = (
            self.current_readonly_metal_mesh_payload_reasons
        )
        previous_readonly_raw_buffer_parameters = (
            self.current_readonly_raw_buffer_parameters
        )
        previous_readonly_metal_parameters = self.current_readonly_metal_parameters
        previous_readonly_metal_parameter_reasons = (
            self.current_readonly_metal_parameter_reasons
        )
        previous_metal_wave_lane_index_parameter = (
            self.current_metal_wave_lane_index_parameter
        )
        previous_metal_wave_lane_count_parameter = (
            self.current_metal_wave_lane_count_parameter
        )
        previous_metal_graphics_builtin_parameter_names = (
            self.current_metal_graphics_builtin_parameter_names
        )
        previous_metal_compute_builtin_parameter_names = (
            self.current_metal_compute_builtin_parameter_names
        )
        self.current_function_name = getattr(func, "name", None)
        self.current_function_return_wrapper = None
        self.current_local_identifier_remaps = {}
        self.current_metal_stage_output_return_active = False
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        (
            self.current_glsl_buffer_block_parameters,
            self.current_glsl_buffer_block_parameter_failures,
            self.current_glsl_buffer_block_parameter_struct_failures,
        ) = self.collect_lowered_glsl_buffer_block_parameters(param_list)
        if shader_type in {"vertex", "fragment", "compute", "ray_generation"}:
            self.validate_stage_parameter_resource_bindings(
                param_list, self.current_function_name
            )
        if shader_type in {"vertex", "fragment"}:
            self.validate_graphics_builtin_parameter_types(param_list, shader_type)
            self.validate_metal_fragment_invocation_density(func, shader_type)
        if shader_type == "compute":
            self.validate_compute_builtin_parameter_types(param_list)
        self.current_unsupported_glsl_buffer_block_parameters = (
            self.collect_unsupported_glsl_buffer_block_parameter_names(param_list)
        )
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        unsupported_metal_ray_function_table_parameter_diagnostics = []
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_metal_mesh_payload_parameter = None
        self.current_metal_mesh_payload_type = None
        self.current_metal_mesh_output_accumulators = {}
        self.current_metal_non_thread_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_reasons = {}
        self.current_readonly_raw_buffer_parameters = set()
        self.current_readonly_metal_parameters = set()
        self.current_readonly_metal_parameter_reasons = {}
        self.current_metal_wave_lane_index_parameter = None
        self.current_metal_wave_lane_count_parameter = None
        self.current_metal_graphics_builtin_parameter_names = {}
        self.current_metal_compute_builtin_parameter_names = {}
        self.local_variable_types = {}
        self.current_address_space_variables = {}
        if shader_type in {"vertex", "fragment"}:
            for interface in self.spirv_stage_input_layouts(shader_type):
                name = getattr(interface, "variable_name", None)
                raw_type = getattr(interface, "data_type", None)
                if not name or not raw_type or "[" in str(name):
                    continue
                mapped_type = self.map_type(raw_type)
                if shader_type == "vertex":
                    vertex_stage_input_parameters.append(
                        {
                            "name": name,
                            "raw_type": raw_type,
                            "mapped_type": mapped_type,
                            "attribute": self.spirv_stage_input_attribute(interface),
                            "semantic": self.spirv_stage_input_semantic(interface),
                        }
                    )
                    self.local_variable_types[name] = self.type_name_string(raw_type)

            for interface in self.spirv_stage_output_layouts(shader_type):
                name = getattr(interface, "variable_name", None)
                raw_type = getattr(interface, "data_type", None)
                if not name or not raw_type or "[" in str(name):
                    continue
                stage_output_parameters.append(
                    {
                        "name": name,
                        "raw_type": raw_type,
                        "mapped_type": self.map_type(raw_type),
                        "semantic": self.spirv_stage_output_semantic(interface),
                        "attribute": self.spirv_stage_output_attribute(interface),
                    }
                )
                self.local_variable_types[name] = self.type_name_string(raw_type)
        skipped_parameter_indices = self.skipped_function_parameter_indices(
            self.current_function_name
        )
        for index, p in enumerate(param_list):
            if index in skipped_parameter_indices:
                continue
            if hasattr(p, "param_type"):
                raw_param_type = (
                    self.type_name_string(p.param_type)
                    if getattr(p.param_type, "generic_args", None)
                    else p.param_type
                )
            elif hasattr(p, "vtype"):
                raw_param_type = p.vtype
            else:
                raw_param_type = "float"
            self.local_variable_types[p.name] = self.type_name_string(raw_param_type)
            address_space = self.parameter_variable_address_space(
                raw_param_type, p, shader_type
            )
            if address_space is not None:
                self.current_address_space_variables[p.name] = address_space
            if self.is_readonly_raw_buffer_parameter(raw_param_type, p, shader_type):
                self.current_readonly_raw_buffer_parameters.add(p.name)
            readonly_parameter_reason = self.readonly_metal_parameter_reason(
                raw_param_type, p, shader_type
            )
            if readonly_parameter_reason is not None:
                self.current_readonly_metal_parameters.add(p.name)
                self.current_readonly_metal_parameter_reasons[p.name] = (
                    readonly_parameter_reason
                )

            if self.is_metal_mesh_output_parameter(shader_type, p):
                continue

            table_array_kind = self.metal_ray_function_table_array_parameter_kind(
                raw_param_type, p
            )
            if table_array_kind is not None:
                self.unsupported_metal_ray_function_table_array_variables[p.name] = (
                    table_array_kind
                )
                unsupported_metal_ray_function_table_parameter_diagnostics.append(
                    self.unsupported_metal_ray_function_table_array_diagnostic(
                        table_array_kind, p.name
                    ).rstrip()
                )
                continue

            acceleration_structure_array = (
                self.metal_acceleration_structure_array_parameter_kind(
                    raw_param_type, p
                )
            )
            if acceleration_structure_array is not None:
                self.unsupported_metal_acceleration_structure_array_variables[
                    p.name
                ] = acceleration_structure_array
                unsupported_metal_ray_function_table_parameter_diagnostics.append(
                    self.unsupported_metal_acceleration_structure_array_diagnostic(
                        p.name
                    ).rstrip()
                )
                continue

            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
                resource_array = self.resource_array_parameter(raw_param_type, p)
                if resource_array is not None:
                    _, array_size = resource_array
                    sampler_parameter_array_sizes[p.name] = array_size
            elif self.is_texture_or_image_resource_type(raw_param_type):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                texture_parameter_raw_types[p.name] = self.resource_base_type(
                    raw_param_type
                )
                resource_array = self.resource_array_parameter(raw_param_type, p)
                if resource_array is not None:
                    _, array_size = resource_array
                    texture_parameter_array_sizes[p.name] = array_size
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=image_format_parameters,
                )
            param_type = self.map_resource_type_with_format(raw_param_type, p)

            semantic = self.semantic_from_node(p)
            if self.is_graphics_stage_output_parameter(p, shader_type):
                stage_output_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "semantic": semantic,
                        "attribute": self.metal_stage_output_parameter_attribute(
                            p, shader_type
                        ),
                    }
                )
                continue
            if self.is_metal_mesh_payload_parameter(shader_type, p):
                self.current_metal_mesh_payload_parameter = p.name
                self.current_metal_mesh_payload_type = self.map_type(raw_param_type)
                self.current_address_space_variables[p.name] = "object_data"
                readonly_reason = self.readonly_metal_mesh_payload_parameter_reason(
                    shader_type, p
                )
                if readonly_reason is not None:
                    self.current_readonly_metal_mesh_payload_parameters.add(p.name)
                    self.current_readonly_metal_mesh_payload_reasons[p.name] = (
                        readonly_reason
                    )
            ray_payload_declaration = self.metal_ray_payload_parameter_declaration(
                param_type, p.name, p, shader_type
            )
            if ray_payload_declaration is not None:
                self.current_metal_non_thread_payload_parameters.add(p.name)
                ray_payload_address_space = self.metal_ray_payload_address_space(
                    shader_type
                )
                if ray_payload_address_space is not None:
                    self.current_address_space_variables[p.name] = (
                        ray_payload_address_space
                    )

            builtin_lowering = self.metal_graphics_builtin_parameter_lowering(
                raw_param_type,
                param_type,
                p,
                shader_type,
                reserved_parameter_names
                | set(params)
                | self.metal_function_local_variable_names(func),
            )
            if builtin_lowering is not None:
                params.append(
                    f"{builtin_lowering['type']} {builtin_lowering['name']} "
                    f"[[{builtin_lowering['attribute']}]]"
                )
                reserved_parameter_names.add(builtin_lowering["name"])
                stage_builtin_parameter_alias_declarations.append(
                    f"{param_type} {p.name} = {param_type}({builtin_lowering['name']});"
                )
                continue

            if self.is_plain_metal_vertex_stage_input_parameter(
                raw_param_type, param_type, p, shader_type
            ):
                vertex_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "attribute": self.metal_vertex_input_location_attribute(p),
                        "semantic": semantic,
                    }
                )
                continue
            if self.is_plain_metal_fragment_stage_input_parameter(
                raw_param_type, param_type, p, shader_type
            ):
                fragment_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "attribute": self.metal_fragment_input_semantic_attribute(
                            semantic
                        ),
                        "semantic": semantic,
                        "preserve_user_attribute": True,
                    }
                )
                continue

            param_attr = (
                self.metal_vertex_input_location_attribute(p)
                if shader_type == "vertex"
                else None
            )
            if param_attr is None:
                param_attr = self.parameter_attribute(
                    raw_param_type, semantic, shader_type, p
                )
            declaration = self.format_parameter_declaration(
                raw_param_type, param_type, p.name, p, shader_type
            )
            if self.should_wrap_metal_vertex_stage_input_parameter(
                raw_param_type, shader_type, p
            ):
                vertex_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "declaration": declaration,
                        "attribute": param_attr,
                        "semantic": semantic,
                    }
                )
                continue
            params.append(f"{declaration}{param_attr}")
            if self.structured_buffer_parameter_requires_length(
                self.current_function_name, p.name
            ):
                length_name = self.structured_buffer_length_parameter_name(p.name)
                self.current_structured_buffer_length_parameters[p.name] = length_name
                array_size = self.structured_buffer_parameter_array_size(
                    raw_param_type, p
                )
                params.append(
                    self.format_structured_buffer_length_parameter(p.name, array_size)
                )
            if self.structured_buffer_requires_counter(raw_param_type):
                counter_name = self.structured_buffer_counter_parameter_name(p.name)
                self.current_structured_buffer_counter_parameters[p.name] = counter_name
                array_size = self.structured_buffer_parameter_array_size(
                    raw_param_type, p
                )
                params.append(
                    self.format_structured_buffer_counter_parameter(p.name, array_size)
                )

        if shader_type == "vertex":
            explicit_stage_builtins = self.explicit_graphics_stage_builtin_parameters(
                param_list, shader_type
            )
            if explicit_stage_builtins.get("vertex_id"):
                self.current_metal_graphics_builtin_parameter_names[
                    "gl_VertexIndex"
                ] = explicit_stage_builtins["vertex_id"]
            reserved_builtin_names = set(reserved_parameter_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for (
                builtin_name,
                name,
                param_type,
                attribute,
            ) in self.required_metal_vertex_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                params.append(f"{param_type} {name} [[{attribute}]]")
                reserved_parameter_names.add(name)
                self.current_metal_graphics_builtin_parameter_names[builtin_name] = name

        if shader_type == "fragment":
            explicit_stage_builtins = self.explicit_graphics_stage_builtin_parameters(
                param_list, shader_type
            )
            if explicit_stage_builtins.get("position"):
                self.current_metal_graphics_builtin_parameter_names["gl_FragCoord"] = (
                    explicit_stage_builtins["position"]
                )
            reserved_builtin_names = set(reserved_parameter_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for (
                builtin_name,
                name,
                param_type,
                attribute,
            ) in self.required_metal_fragment_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                params.append(f"{param_type} {name} [[{attribute}]]")
                reserved_parameter_names.add(name)
                self.current_metal_graphics_builtin_parameter_names[builtin_name] = name

        if shader_type == "compute":
            existing_param_names = {getattr(p, "name", None) for p in param_list}
            explicit_stage_builtins = self.explicit_compute_stage_builtin_parameters(
                param_list
            )
            self.current_metal_compute_builtin_parameter_names.update(
                {
                    builtin_name: explicit_stage_builtins[attribute]
                    for builtin_name, _param_type, attribute in (
                        self.compute_builtin_parameter_specs()
                    )
                    if attribute in explicit_stage_builtins
                    and explicit_stage_builtins[attribute]
                }
            )
            self.current_metal_wave_lane_index_parameter = explicit_stage_builtins.get(
                "thread_index_in_simdgroup"
            )
            self.current_metal_wave_lane_count_parameter = explicit_stage_builtins.get(
                "threads_per_simdgroup"
            )
            reserved_builtin_names = set(existing_param_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for name, param_type, attribute in self.required_compute_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                if name not in existing_param_names:
                    params.append(f"{param_type} {name} [[{attribute}]]")
                    reserved_parameter_names.add(name)
                builtin_name = self.compute_builtin_name_for_metal_attribute(attribute)
                if builtin_name is not None:
                    self.current_metal_compute_builtin_parameter_names[builtin_name] = (
                        name
                    )
                if attribute == "thread_index_in_simdgroup":
                    self.current_metal_wave_lane_index_parameter = name
                elif attribute == "threads_per_simdgroup":
                    self.current_metal_wave_lane_count_parameter = name

        reserved_parameter_names.update(self.global_resource_parameter_names())
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            self.cbuffer_variables, reserved_names=reserved_parameter_names
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            self.cbuffer_variables
        )
        self.current_local_identifier_remaps = (
            self.collect_metal_local_identifier_remaps(
                func,
                param_list,
                vertex_stage_input_parameters
                + fragment_stage_input_parameters
                + stage_output_parameters,
                stage_local_variables,
            )
        )

        params_str = ", ".join(params)
        if shader_type is None:
            params_str = self.append_required_stage_output_parameter(
                params_str, self.current_function_name
            )
            params_str = self.append_required_stage_parameter_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_cbuffer_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_global_resource_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_metal_mesh_dispatch_context_parameters(
                params_str,
                func,
                reserved_parameter_names,
            )
            params_str = self.append_required_metal_wave_lane_parameters(
                params_str, self.current_function_name
            )
            for parameter in self.required_function_stage_parameters(
                self.current_function_name
            ):
                self.local_variable_types[parameter.name] = self.type_name_string(
                    self.parameter_raw_type(parameter)
                )
            stage_output_type = self.required_function_stage_output_type(
                self.current_function_name
            )
            if stage_output_type is not None:
                self.local_variable_types["output"] = self.type_name_string(
                    stage_output_type
                )

        if hasattr(func, "return_type"):
            raw_return_type = self.type_name_string(func.return_type)
            if "[" in raw_return_type:
                raise ValueError(
                    "Metal output does not support C-style array return types; "
                    "wrap the array in a struct or use an output parameter"
                )
            return_type = self.map_type(raw_return_type)
        else:
            raw_return_type = "void"
            return_type = "void"
        if shader_type in {"vertex", "fragment"}:
            self.validate_function_return_semantic_type(
                func, raw_return_type, shader_type
            )
            if stage_output_parameters and return_type != "void":
                raise ValueError(
                    f"Metal {shader_type} function '{self.current_function_name}' "
                    "cannot combine output parameters with a non-void return type"
                )
        self.current_function_return_type = raw_return_type
        self.validate_metal_ray_stage_signature(func, shader_type, raw_return_type)
        parameter_diagnostics = self.glsl_buffer_block_parameter_diagnostics(
            "Metal", param_list, indent
        )
        if parameter_diagnostics:
            code += parameter_diagnostics
            code += "  " * indent
        unsupported_function_reason = self.unsupported_glsl_buffer_block_functions.get(
            getattr(func, "name", None)
        )
        if unsupported_function_reason is not None:
            code += self.unsupported_glsl_buffer_block_function_placeholder(
                "Metal", getattr(func, "name", None), unsupported_function_reason
            )
            self.current_function_name = previous_function_name
            self.current_function_return_type = previous_function_return_type
            self.current_function_return_wrapper = previous_function_return_wrapper
            self.current_local_identifier_remaps = previous_local_identifier_remaps
            self.local_variable_types = previous_local_variable_types
            self.current_address_space_variables = previous_address_space_variables
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.cbuffer_parameter_names = previous_cbuffer_parameter_names
            self.cbuffer_member_references = previous_cbuffer_member_references
            self.ambiguous_cbuffer_members = previous_ambiguous_cbuffer_members
            self.current_glsl_buffer_block_parameters = (
                previous_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_parameters = (
                previous_unsupported_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_glsl_buffer_block_local_variables
            )
            self.unsupported_metal_ray_function_table_array_variables = (
                previous_unsupported_metal_ray_function_table_array_variables
            )
            self.unsupported_metal_acceleration_structure_array_variables = (
                previous_unsupported_metal_acceleration_structure_array_variables
            )
            self.current_glsl_buffer_block_parameter_failures = (
                previous_glsl_buffer_block_parameter_failures
            )
            self.current_glsl_buffer_block_parameter_struct_failures = (
                previous_glsl_buffer_block_parameter_struct_failures
            )
            self.current_structured_buffer_length_parameters = (
                previous_structured_buffer_length_parameters
            )
            self.current_structured_buffer_counter_parameters = (
                previous_structured_buffer_counter_parameters
            )
            self.current_metal_mesh_output_config = previous_metal_mesh_output_config
            self.current_metal_mesh_output_parameter = (
                previous_metal_mesh_output_parameter
            )
            self.current_metal_mesh_grid_properties_parameter = (
                previous_metal_mesh_grid_properties_parameter
            )
            self.current_metal_mesh_payload_parameter = (
                previous_metal_mesh_payload_parameter
            )
            self.current_metal_mesh_payload_type = previous_metal_mesh_payload_type
            self.current_metal_mesh_output_accumulators = (
                previous_metal_mesh_output_accumulators
            )
            self.current_metal_non_thread_payload_parameters = (
                previous_metal_non_thread_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_parameters = (
                previous_readonly_metal_mesh_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_reasons = (
                previous_readonly_metal_mesh_payload_reasons
            )
            self.current_readonly_raw_buffer_parameters = (
                previous_readonly_raw_buffer_parameters
            )
            self.current_readonly_metal_parameters = previous_readonly_metal_parameters
            self.current_readonly_metal_parameter_reasons = (
                previous_readonly_metal_parameter_reasons
            )
            self.current_metal_wave_lane_index_parameter = (
                previous_metal_wave_lane_index_parameter
            )
            self.current_metal_wave_lane_count_parameter = (
                previous_metal_wave_lane_count_parameter
            )
            self.current_metal_graphics_builtin_parameter_names = (
                previous_metal_graphics_builtin_parameter_names
            )
            self.current_metal_compute_builtin_parameter_names = (
                previous_metal_compute_builtin_parameter_names
            )
            return code

        body = getattr(func, "body", None)
        if shader_type is None and body is None:
            semantic = self.semantic_from_node(func)
            function_name = entry_name or func.name
            semantic_attr = self.map_non_stage_function_semantic(semantic)
            code += f"{return_type} {function_name}({params_str}){semantic_attr};\n\n"
            self.current_function_name = previous_function_name
            self.current_function_return_type = previous_function_return_type
            self.current_function_return_wrapper = previous_function_return_wrapper
            self.current_local_identifier_remaps = previous_local_identifier_remaps
            self.local_variable_types = previous_local_variable_types
            self.current_address_space_variables = previous_address_space_variables
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.cbuffer_parameter_names = previous_cbuffer_parameter_names
            self.cbuffer_member_references = previous_cbuffer_member_references
            self.ambiguous_cbuffer_members = previous_ambiguous_cbuffer_members
            self.current_glsl_buffer_block_parameters = (
                previous_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_parameters = (
                previous_unsupported_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_glsl_buffer_block_local_variables
            )
            self.unsupported_metal_ray_function_table_array_variables = (
                previous_unsupported_metal_ray_function_table_array_variables
            )
            self.unsupported_metal_acceleration_structure_array_variables = (
                previous_unsupported_metal_acceleration_structure_array_variables
            )
            self.current_glsl_buffer_block_parameter_failures = (
                previous_glsl_buffer_block_parameter_failures
            )
            self.current_glsl_buffer_block_parameter_struct_failures = (
                previous_glsl_buffer_block_parameter_struct_failures
            )
            self.current_structured_buffer_length_parameters = (
                previous_structured_buffer_length_parameters
            )
            self.current_structured_buffer_counter_parameters = (
                previous_structured_buffer_counter_parameters
            )
            self.current_metal_mesh_output_config = previous_metal_mesh_output_config
            self.current_metal_mesh_output_parameter = (
                previous_metal_mesh_output_parameter
            )
            self.current_metal_mesh_grid_properties_parameter = (
                previous_metal_mesh_grid_properties_parameter
            )
            self.current_metal_mesh_payload_parameter = (
                previous_metal_mesh_payload_parameter
            )
            self.current_metal_mesh_payload_type = previous_metal_mesh_payload_type
            self.current_metal_mesh_output_accumulators = (
                previous_metal_mesh_output_accumulators
            )
            self.current_metal_non_thread_payload_parameters = (
                previous_metal_non_thread_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_parameters = (
                previous_readonly_metal_mesh_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_reasons = (
                previous_readonly_metal_mesh_payload_reasons
            )
            self.current_readonly_raw_buffer_parameters = (
                previous_readonly_raw_buffer_parameters
            )
            self.current_readonly_metal_parameters = previous_readonly_metal_parameters
            self.current_readonly_metal_parameter_reasons = (
                previous_readonly_metal_parameter_reasons
            )
            self.current_metal_wave_lane_index_parameter = (
                previous_metal_wave_lane_index_parameter
            )
            self.current_metal_wave_lane_count_parameter = (
                previous_metal_wave_lane_count_parameter
            )
            self.current_metal_graphics_builtin_parameter_names = (
                previous_metal_graphics_builtin_parameter_names
            )
            self.current_metal_compute_builtin_parameter_names = (
                previous_metal_compute_builtin_parameter_names
            )
            return code

        if shader_type == "vertex":
            function_name = entry_name or f"vertex_{func.name}"
            if vertex_stage_input_parameters:
                stage_input_struct_name = self.unique_vertex_stage_input_struct_name(
                    function_name
                )
                stage_input_parameter_name = self.unique_metal_generated_name(
                    "_crossglInput",
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                self.local_variable_types[stage_input_parameter_name] = (
                    stage_input_struct_name
                )
                code += self.generate_metal_vertex_stage_input_parameter_struct(
                    stage_input_struct_name, vertex_stage_input_parameters
                )
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"{stage_input_struct_name} {stage_input_parameter_name} "
                    "[[stage_in]]",
                )
                vertex_stage_input_alias_declarations = (
                    self.generate_metal_vertex_stage_input_alias_declarations(
                        stage_input_parameter_name, vertex_stage_input_parameters
                    )
                )
            params_str = self.append_global_resource_parameters(
                params_str,
                self.current_function_name,
                func,
                filter_writable_textures=True,
            )
            if stage_output_parameters:
                stage_output_struct_name = self.unique_return_wrapper_struct_name(
                    function_name
                )
                code += self.generate_metal_stage_output_parameter_struct(
                    stage_output_struct_name, stage_output_parameters
                )
                return_type = stage_output_struct_name
                self.current_function_return_wrapper = None
            else:
                return_wrapper = self.function_return_semantic_wrapper(
                    func, raw_return_type, return_type, shader_type, function_name
                )
                if return_wrapper is not None:
                    code += self.generate_return_wrapper_struct(return_wrapper)
                    return_type = return_wrapper["struct_name"]
                self.current_function_return_wrapper = return_wrapper
            code += f"vertex {return_type} {function_name}({params_str}) {{\n"
        elif shader_type == "fragment":
            function_name = entry_name or f"fragment_{func.name}"
            if fragment_stage_input_parameters:
                stage_input_struct_name = self.unique_vertex_stage_input_struct_name(
                    function_name
                )
                stage_input_parameter_name = self.unique_metal_generated_name(
                    "_crossglInput",
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                self.local_variable_types[stage_input_parameter_name] = (
                    stage_input_struct_name
                )
                code += self.generate_metal_vertex_stage_input_parameter_struct(
                    stage_input_struct_name, fragment_stage_input_parameters
                )
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"{stage_input_struct_name} {stage_input_parameter_name} "
                    "[[stage_in]]",
                )
                fragment_stage_input_alias_declarations = (
                    self.generate_metal_vertex_stage_input_alias_declarations(
                        stage_input_parameter_name, fragment_stage_input_parameters
                    )
                )
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            if stage_output_parameters:
                stage_output_struct_name = self.unique_return_wrapper_struct_name(
                    function_name
                )
                code += self.generate_metal_stage_output_parameter_struct(
                    stage_output_struct_name, stage_output_parameters
                )
                return_type = stage_output_struct_name
                self.current_function_return_wrapper = None
            else:
                return_wrapper = self.function_return_semantic_wrapper(
                    func, raw_return_type, return_type, shader_type, function_name
                )
                if return_wrapper is not None:
                    code += self.generate_return_wrapper_struct(return_wrapper)
                    return_type = return_wrapper["struct_name"]
                self.current_function_return_wrapper = return_wrapper
            code += f"fragment {return_type} {function_name}({params_str}) {{\n"
        elif shader_type == "geometry":
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            function_name = entry_name or f"geometry_{func.name}"
            self.validate_metal_geometry_stage(
                func,
                param_list,
                stage_node=stage_node,
            )
            code += self.generate_metal_geometry_stage_comments(
                func,
                param_list,
                stage_node=stage_node,
            )
            code += f"{return_type} {function_name}({params_str}) {{\n"
        elif shader_type in {"tessellation_control", "tessellation_evaluation"}:
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            function_name = entry_name or self.stage_entry_base_name(shader_type, func)
            self.validate_metal_tessellation_stage(func, param_list, shader_type)
            code += self.generate_metal_tessellation_stage_comments(
                func,
                param_list,
                shader_type,
            )
            code += f"{return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["compute", "ray_generation"]:
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            function_name = entry_name or self.stage_entry_base_name(shader_type, func)
            self.validate_metal_kernel_return_type(func, shader_type, raw_return_type)
            code += f"kernel {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["mesh", "object", "task", "amplification"]:
            stage_keyword = "mesh" if shader_type == "mesh" else "object"
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            stage_attribute = self.metal_mesh_stage_attribute(
                shader_type, func, execution_config
            )
            generated_mesh_payload_parameter = (
                self.generated_metal_mesh_payload_parameter(
                    shader_type, func, reserved_parameter_names
                )
            )
            if generated_mesh_payload_parameter is not None:
                payload_type, payload_name = generated_mesh_payload_parameter
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"object_data {payload_type}& {payload_name} [[payload]]",
                )
                self.current_metal_mesh_payload_parameter = payload_name
                self.current_metal_mesh_payload_type = payload_type
                self.current_address_space_variables[payload_name] = "object_data"
            mesh_grid_properties_parameter = self.metal_mesh_grid_properties_parameter(
                shader_type, func, reserved_parameter_names
            )
            if mesh_grid_properties_parameter is not None:
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"mesh_grid_properties {mesh_grid_properties_parameter}",
                )
                self.current_metal_mesh_grid_properties_parameter = (
                    mesh_grid_properties_parameter
                )
            mesh_output = self.metal_mesh_stage_output_config(
                shader_type, func, function_name, reserved_parameter_names
            )
            if mesh_output is not None:
                if mesh_output.get("generated_vertex_struct", False):
                    code += self.generate_metal_mesh_vertex_output_struct(mesh_output)
                params_str = self.append_parameter_declaration(
                    params_str,
                    self.metal_mesh_stage_output_parameter_declaration(mesh_output),
                )
                self.current_metal_mesh_output_parameter = mesh_output["parameter_name"]
                self.current_metal_mesh_output_config = mesh_output
                self.validate_metal_mesh_output_usage(func)
            code += (
                f"{stage_attribute} {return_type} {function_name}({params_str}) {{\n"
            )
        elif shader_type in [
            "ray_any_hit",
            "ray_closest_hit",
            "ray_miss",
            "ray_callable",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        ]:
            rt_stage_map = {
                "ray_any_hit": "anyhit",
                "ray_closest_hit": "closesthit",
                "ray_miss": "miss",
                "ray_callable": "callable",
                "anyhit": "anyhit",
                "closesthit": "closesthit",
                "miss": "miss",
                "callable": "callable",
            }
            stage_keyword = rt_stage_map.get(shader_type, shader_type)
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            code += f"[[visible]] {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["ray_intersection", "intersection"]:
            function_name = entry_name or f"intersection_{func.name}"
            stage_attribute = self.metal_ray_intersection_stage_attribute(
                func, raw_return_type
            )
            code += (
                f"{stage_attribute} {return_type} "
                f"{function_name}({params_str}) {{\n"
            )
        else:
            semantic = self.semantic_from_node(func)
            function_name = entry_name or func.name
            semantic_attr = self.map_non_stage_function_semantic(semantic)
            code += f"{return_type} {function_name}({params_str}){semantic_attr} {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_sampler_parameter_array_sizes = (
            self.current_sampler_parameter_array_sizes
        )
        previous_texture_parameters = self.current_texture_parameters
        previous_texture_parameter_raw_types = self.current_texture_parameter_raw_types
        previous_texture_parameter_array_sizes = (
            self.current_texture_parameter_array_sizes
        )
        previous_texture_alias_sources = self.current_texture_alias_sources
        previous_image_format_parameters = self.current_image_format_parameters
        self.current_sampler_parameters = sampler_parameters
        self.current_sampler_parameter_array_sizes = sampler_parameter_array_sizes
        self.current_texture_parameters = texture_parameters
        self.current_texture_parameter_raw_types = texture_parameter_raw_types
        self.current_texture_parameter_array_sizes = texture_parameter_array_sizes
        self.current_texture_alias_sources = {}
        self.current_image_format_parameters = image_format_parameters
        self.register_metal_buffer_resource_parameter_scope(
            self.current_function_name,
            include_all=shader_type
            in {
                "vertex",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "compute",
                "ray_generation",
            },
        )
        if shader_type == "mesh" and self.current_metal_mesh_output_config is not None:
            self.current_metal_mesh_output_accumulators = (
                self.collect_metal_mesh_output_accumulators(
                    func, reserved_parameter_names
                )
            )
            code += self.generate_metal_mesh_output_accumulator_declarations()
        for diagnostic in unsupported_metal_ray_function_table_parameter_diagnostics:
            code += f"    {diagnostic}\n"
        for declaration in stage_builtin_parameter_alias_declarations:
            code += f"    {declaration}\n"
        for declaration in vertex_stage_input_alias_declarations:
            code += f"    {declaration}\n"
        for declaration in fragment_stage_input_alias_declarations:
            code += f"    {declaration}\n"
        code += self.generate_metal_stage_output_parameter_locals(
            stage_output_parameters
        )
        self.current_metal_stage_output_return_active = bool(stage_output_parameters)
        stage_value_variables = [
            local_var
            for local_var in stage_local_variables or []
            if not self.is_stage_local_resource_variable(local_var)
        ]
        code += self.generate_statement_body(stage_value_variables, 1)
        code += self.generate_statement_body(body, 1)
        if stage_output_parameters:
            code += self.generate_metal_stage_output_parameter_return(
                stage_output_struct_name,
                stage_output_parameters,
                reserved_parameter_names
                | self.metal_function_local_variable_names(func),
            )
        elif self.metal_needs_fallback_return(body, return_type):
            fallback = self.metal_default_value_expression(raw_return_type)
            code += (
                f"    return {fallback} "
                "/* fallback for unmatched generated control flow */;\n"
            )
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_sampler_parameter_array_sizes = (
            previous_sampler_parameter_array_sizes
        )
        self.current_texture_parameters = previous_texture_parameters
        self.current_texture_parameter_raw_types = previous_texture_parameter_raw_types
        self.current_texture_parameter_array_sizes = (
            previous_texture_parameter_array_sizes
        )
        self.current_texture_alias_sources = previous_texture_alias_sources
        self.current_image_format_parameters = previous_image_format_parameters
        self.current_structured_buffer_length_parameters = (
            previous_structured_buffer_length_parameters
        )
        self.current_structured_buffer_counter_parameters = (
            previous_structured_buffer_counter_parameters
        )
        self.current_metal_mesh_output_config = previous_metal_mesh_output_config
        self.current_metal_mesh_output_parameter = previous_metal_mesh_output_parameter
        self.current_metal_mesh_grid_properties_parameter = (
            previous_metal_mesh_grid_properties_parameter
        )
        self.current_metal_mesh_payload_parameter = (
            previous_metal_mesh_payload_parameter
        )
        self.current_metal_mesh_payload_type = previous_metal_mesh_payload_type
        self.current_metal_mesh_output_accumulators = (
            previous_metal_mesh_output_accumulators
        )
        self.current_metal_non_thread_payload_parameters = (
            previous_metal_non_thread_payload_parameters
        )
        self.current_readonly_metal_mesh_payload_parameters = (
            previous_readonly_metal_mesh_payload_parameters
        )
        self.current_readonly_metal_mesh_payload_reasons = (
            previous_readonly_metal_mesh_payload_reasons
        )
        self.current_readonly_raw_buffer_parameters = (
            previous_readonly_raw_buffer_parameters
        )
        self.current_readonly_metal_parameters = previous_readonly_metal_parameters
        self.current_readonly_metal_parameter_reasons = (
            previous_readonly_metal_parameter_reasons
        )
        self.current_metal_wave_lane_index_parameter = (
            previous_metal_wave_lane_index_parameter
        )
        self.current_metal_wave_lane_count_parameter = (
            previous_metal_wave_lane_count_parameter
        )
        self.current_metal_graphics_builtin_parameter_names = (
            previous_metal_graphics_builtin_parameter_names
        )
        self.current_function_name = previous_function_name
        self.current_function_return_type = previous_function_return_type
        self.current_function_return_wrapper = previous_function_return_wrapper
        self.current_local_identifier_remaps = previous_local_identifier_remaps
        self.current_metal_stage_output_return_active = (
            previous_stage_output_return_active
        )
        self.local_variable_types = previous_local_variable_types
        self.current_address_space_variables = previous_address_space_variables
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.cbuffer_parameter_names = previous_cbuffer_parameter_names
        self.cbuffer_member_references = previous_cbuffer_member_references
        self.ambiguous_cbuffer_members = previous_ambiguous_cbuffer_members
        self.current_glsl_buffer_block_parameters = (
            previous_glsl_buffer_block_parameters
        )
        self.current_unsupported_glsl_buffer_block_parameters = (
            previous_unsupported_glsl_buffer_block_parameters
        )
        self.current_unsupported_glsl_buffer_block_local_variables = (
            previous_unsupported_glsl_buffer_block_local_variables
        )
        self.unsupported_metal_ray_function_table_array_variables = (
            previous_unsupported_metal_ray_function_table_array_variables
        )
        self.unsupported_metal_acceleration_structure_array_variables = (
            previous_unsupported_metal_acceleration_structure_array_variables
        )
        self.current_glsl_buffer_block_parameter_failures = (
            previous_glsl_buffer_block_parameter_failures
        )
        self.current_glsl_buffer_block_parameter_struct_failures = (
            previous_glsl_buffer_block_parameter_struct_failures
        )
        self.current_metal_compute_builtin_parameter_names = (
            previous_metal_compute_builtin_parameter_names
        )

        code += "}\n\n"
        return code

    def metal_needs_fallback_return(self, body, return_type):
        if return_type == "void":
            return False
        if hasattr(body, "statements"):
            statements = list(getattr(body, "statements", []) or [])
        elif isinstance(body, list):
            statements = body
        else:
            statements = []
        if not statements:
            return True
        return not isinstance(statements[-1], (ReturnNode, BackendReturnNode))

    def compute_builtin_parameter_specs(self):
        return [
            ("gl_GlobalInvocationID", "uint3", "thread_position_in_grid"),
            ("gl_LocalInvocationID", "uint3", "thread_position_in_threadgroup"),
            ("gl_WorkGroupID", "uint3", "threadgroup_position_in_grid"),
            ("gl_LocalInvocationIndex", "uint", "thread_index_in_threadgroup"),
            ("gl_WorkGroupSize", "uint3", "threads_per_threadgroup"),
            ("gl_NumWorkGroups", "uint3", "threadgroups_per_grid"),
            ("gl_SubgroupInvocationID", "uint", "thread_index_in_simdgroup"),
            ("gl_SubgroupSize", "uint", "threads_per_simdgroup"),
        ]

    def compute_builtin_name_for_metal_attribute(self, attribute):
        for (
            builtin_name,
            _param_type,
            metal_attribute,
        ) in self.compute_builtin_parameter_specs():
            if metal_attribute == attribute:
                return builtin_name
        return None

    def metal_compute_builtin_expression_name(self, name):
        if name is None:
            return name
        return self.current_metal_compute_builtin_parameter_names.get(name, name)

    def metal_glsl_subgroup_lane_builtin_diagnostic(self, name):
        if name in self.local_variable_types:
            return None
        builtin = self.METAL_GLSL_SUBGROUP_LANE_BUILTINS.get(name)
        if builtin is None:
            return None
        attribute, result_type = builtin
        fallback = self.diagnostic_zero_value_for_type(result_type)
        return (
            f"{fallback} /* unsupported Metal GLSL subgroup builtin: {name} "
            f"requires compute-stage {attribute} value */"
        )

    def metal_graphics_builtin_expression_name(self, name):
        if name is None or name in self.local_variable_types:
            return name
        return self.current_metal_graphics_builtin_parameter_names.get(name, name)

    def metal_builtin_expression_name(self, name):
        graphics_name = self.metal_graphics_builtin_expression_name(name)
        if graphics_name != name:
            return graphics_name
        compute_name = self.metal_compute_builtin_expression_name(name)
        if compute_name != name:
            return compute_name
        diagnostic = self.metal_glsl_subgroup_lane_builtin_diagnostic(name)
        if diagnostic is not None:
            return diagnostic
        return name

    def metal_graphics_builtin_result_type(self, name):
        if (
            name is None
            or name in self.local_variable_types
            or name not in self.current_metal_graphics_builtin_parameter_names
        ):
            return None
        if name == "gl_VertexIndex":
            return "uint"
        if name == "gl_FragCoord":
            return "float4"
        return None

    def metal_compute_builtin_result_type(self, name):
        if name is None:
            return None
        if name in self.current_metal_compute_builtin_parameter_names:
            for (
                builtin_name,
                param_type,
                _attribute,
            ) in self.compute_builtin_parameter_specs():
                if builtin_name == name:
                    return param_type
        return None

    def metal_compute_builtin_parameter_base_name(self, builtin_name, attribute):
        return {
            "gl_GlobalInvocationID": "thread_position_in_grid",
            "gl_LocalInvocationID": "thread_position_in_threadgroup",
            "gl_WorkGroupID": "threadgroup_position_in_grid",
            "gl_LocalInvocationIndex": "thread_index_in_threadgroup",
            "gl_WorkGroupSize": "threads_per_threadgroup",
            "gl_NumWorkGroups": "threadgroups_per_grid",
            "gl_SubgroupInvocationID": "thread_index_in_simdgroup",
            "gl_SubgroupSize": "threads_per_simdgroup",
        }.get(builtin_name, attribute)

    def required_compute_builtin_parameters(
        self, func, reserved_names=None, explicit_stage_builtins=None
    ):
        used_names = self.used_compute_builtin_names(getattr(func, "body", []))
        used_wave_operations = self.used_wave_stage_builtin_operations(
            getattr(func, "body", [])
        )
        func_name = getattr(func, "name", None)
        if func_name:
            used_wave_operations.update(
                self.function_metal_wave_lane_dependencies.get(func_name, set())
            )
        reserved_names = set(reserved_names or ())
        explicit_stage_builtins = explicit_stage_builtins or {}
        required_parameters = []
        for (
            builtin_name,
            param_type,
            attribute,
        ) in self.compute_builtin_parameter_specs():
            if builtin_name not in used_names or attribute in explicit_stage_builtins:
                continue
            name = self.unique_metal_generated_name(
                self.metal_compute_builtin_parameter_base_name(builtin_name, attribute),
                reserved_names,
            )
            reserved_names.add(name)
            required_parameters.append((name, param_type, attribute))
        required_attributes = {
            attribute for _name, _param_type, attribute in required_parameters
        }
        wave_builtin_parameters = [
            (
                "WaveGetLaneIndex",
                "crossglWaveLaneIndex",
                "uint",
                "thread_index_in_simdgroup",
            ),
            (
                "WaveGetLaneCount",
                "crossglWaveLaneCount",
                "uint",
                "threads_per_simdgroup",
            ),
        ]
        for operation, base_name, param_type, attribute in wave_builtin_parameters:
            if operation not in used_wave_operations:
                continue
            if attribute in explicit_stage_builtins or attribute in required_attributes:
                continue
            name = self.unique_metal_generated_name(base_name, reserved_names)
            reserved_names.add(name)
            required_parameters.append((name, param_type, attribute))
            required_attributes.add(attribute)
        return required_parameters

    def used_compute_builtin_names(self, body):
        builtin_names = {
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "gl_SubgroupInvocationID",
            "gl_SubgroupSize",
        }
        used_names = set()
        for node in self.iter_ast_nodes(body):
            if hasattr(node, "__class__") and "Identifier" in str(node.__class__):
                name = getattr(node, "name", "")
                base_name = name.split(".", 1)[0]
                if base_name in builtin_names:
                    used_names.add(base_name)
        return used_names

    def used_wave_stage_builtin_operations(self, body):
        operations = set()
        for node in self.iter_ast_nodes(body):
            if isinstance(node, WaveOpNode):
                operation = getattr(node, "operation", None)
            elif isinstance(node, FunctionCallNode):
                operation = self.function_call_name(node)
            else:
                continue
            if operation == "WaveMatch":
                operations.add("WaveGetLaneCount")
            elif operation in self.METAL_WAVE_MULTI_PREFIX_INTRINSICS:
                operations.add("WaveGetLaneIndex")
                operations.add("WaveGetLaneCount")
            elif operation in {"WaveGetLaneIndex", "WaveGetLaneCount"}:
                operations.add(operation)
        return operations

    def explicit_compute_stage_builtin_parameters(self, parameters):
        stage_parameters = {}
        builtin_attributes = {
            attribute
            for _builtin_name, _param_type, attribute in (
                self.compute_builtin_parameter_specs()
            )
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            if metal_semantic in builtin_attributes or metal_semantic in {
                "thread_index_in_simdgroup",
                "threads_per_simdgroup",
            }:
                stage_parameters[metal_semantic] = getattr(parameter, "name", None)
        return stage_parameters

    def validate_compute_builtin_parameter_types(self, parameters):
        # Metal allows the positional / dimension compute builtins to be declared
        # as a scalar `uint`, `uint2`, or `uint3` (the driver fills the requested
        # component count). See Apple's "Creating threads and threadgroups" guide,
        # which uses the uint2 form. Restricting these to uint3 rejected valid MLX
        # reduction kernels (softmax/rms_norm/layer_norm/... use scalar `uint`).
        positional_builtin_types = ("uint", "uint2", "uint3")
        expected_types = {
            "thread_position_in_grid": positional_builtin_types,
            "thread_position_in_threadgroup": positional_builtin_types,
            "threadgroup_position_in_grid": positional_builtin_types,
            "thread_index_in_threadgroup": "uint",
            "threads_per_threadgroup": positional_builtin_types,
            "threadgroups_per_grid": positional_builtin_types,
            "thread_index_in_simdgroup": "uint",
            "threads_per_simdgroup": "uint",
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = expected_types.get(metal_semantic)
            if expected_type is None:
                continue
            actual_type = self.map_type(self.parameter_raw_type(parameter))
            expected_values = (
                expected_type if isinstance(expected_type, tuple) else (expected_type,)
            )
            if actual_type not in expected_values:
                name = getattr(parameter, "name", "<anonymous>")
                expected_label = (
                    expected_values[0]
                    if len(expected_values) == 1
                    else "one of " + ", ".join(expected_values)
                )
                raise ValueError(
                    f"Metal compute semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires parameter '{name}' to "
                    f"have type {expected_label}, got {actual_type}"
                )

    def collect_function_metal_wave_lane_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.used_wave_stage_builtin_operations(
                getattr(func, "body", [])
            )
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                current_dependencies = dependencies.setdefault(func_name, set())
                before = set(current_dependencies)
                for called_name in calls:
                    current_dependencies.update(dependencies.get(called_name, set()))
                if current_dependencies != before:
                    changed = True
        return dependencies

    def collect_function_metal_wave_lane_parameter_names(self, functions):
        parameter_names = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            dependencies = self.function_metal_wave_lane_dependencies.get(
                func_name, set()
            )
            if not func_name or not dependencies:
                continue
            reserved_names = {
                getattr(parameter, "name", None)
                for parameter in getattr(
                    func, "parameters", getattr(func, "params", [])
                )
                if getattr(parameter, "name", None)
            }
            reserved_names.update(self.metal_function_local_variable_names(func))
            names = {}
            for operation, base_name in (
                ("WaveGetLaneIndex", "crossglWaveLaneIndex"),
                ("WaveGetLaneCount", "crossglWaveLaneCount"),
            ):
                if operation not in dependencies:
                    continue
                name = self.unique_metal_generated_name(base_name, reserved_names)
                reserved_names.add(name)
                names[operation] = name
            parameter_names[func_name] = names
        return parameter_names

    def append_required_metal_wave_lane_parameters(self, params_str, func_name):
        parameter_names = self.function_metal_wave_lane_parameter_names.get(
            func_name, {}
        )
        lane_index_name = parameter_names.get("WaveGetLaneIndex")
        if lane_index_name is not None:
            params_str = self.append_parameter_declaration(
                params_str, f"uint {lane_index_name}"
            )
            self.current_metal_wave_lane_index_parameter = lane_index_name
        lane_count_name = parameter_names.get("WaveGetLaneCount")
        if lane_count_name is not None:
            params_str = self.append_parameter_declaration(
                params_str, f"uint {lane_count_name}"
            )
            self.current_metal_wave_lane_count_parameter = lane_count_name
        return params_str

    def required_metal_wave_lane_context_arguments(self, func_name):
        dependencies = self.function_metal_wave_lane_dependencies.get(func_name, set())
        if not dependencies:
            return []
        args = []
        if "WaveGetLaneIndex" in dependencies:
            if self.current_metal_wave_lane_index_parameter is None:
                return None
            args.append(self.current_metal_wave_lane_index_parameter)
        if "WaveGetLaneCount" in dependencies:
            if self.current_metal_wave_lane_count_parameter is None:
                return None
            args.append(self.current_metal_wave_lane_count_parameter)
        return args

    def metal_wave_lane_helper_call_diagnostic(self, func_name):
        dependencies = self.function_metal_wave_lane_dependencies.get(func_name, set())
        if not dependencies:
            return None
        missing_requirements = []
        if (
            "WaveGetLaneIndex" in dependencies
            and self.current_metal_wave_lane_index_parameter is None
        ):
            missing_requirements.append("thread_index_in_simdgroup")
        if (
            "WaveGetLaneCount" in dependencies
            and self.current_metal_wave_lane_count_parameter is None
        ):
            missing_requirements.append("threads_per_simdgroup")
        if not missing_requirements:
            return None
        requirement = " and ".join(missing_requirements)
        diagnostic = (
            "/* unsupported Metal wave helper call: function "
            f"'{func_name}' requires compute-stage {requirement} value */"
        )
        return_type = self.function_return_types.get(func_name, "void")
        if self.map_type(return_type) == "void":
            return diagnostic
        return f"{self.diagnostic_zero_value_for_type(return_type)} {diagnostic}"

    def metal_vertex_builtin_parameter_specs(self):
        return [
            ("gl_VertexIndex", "uint", "vertex_id"),
        ]

    def metal_fragment_builtin_parameter_specs(self):
        return [
            ("gl_FragCoord", "float4", "position"),
        ]

    def explicit_graphics_stage_builtin_parameters(self, parameters, stage_name):
        if stage_name == "vertex":
            builtin_specs = self.metal_vertex_builtin_parameter_specs()
        elif stage_name == "fragment":
            builtin_specs = self.metal_fragment_builtin_parameter_specs()
        else:
            return {}
        stage_parameters = {}
        builtin_attributes = {
            attribute for _builtin_name, _param_type, attribute in builtin_specs
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            if metal_semantic in builtin_attributes:
                stage_parameters[metal_semantic] = getattr(parameter, "name", None)
        return stage_parameters

    def required_metal_vertex_builtin_parameters(
        self, func, reserved_names=None, explicit_stage_builtins=None
    ):
        used_names = self.used_metal_vertex_builtin_names(getattr(func, "body", []))
        reserved_names = set(reserved_names or ())
        explicit_stage_builtins = explicit_stage_builtins or {}
        required_parameters = []
        for (
            builtin_name,
            param_type,
            attribute,
        ) in self.metal_vertex_builtin_parameter_specs():
            if builtin_name not in used_names or attribute in explicit_stage_builtins:
                continue
            name = self.unique_metal_generated_name("_crossglVertexID", reserved_names)
            reserved_names.add(name)
            required_parameters.append((builtin_name, name, param_type, attribute))
        return required_parameters

    def used_metal_vertex_builtin_names(self, body):
        builtin_names = {"gl_VertexIndex"}
        used_names = set()
        for node in self.iter_ast_nodes(body):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name:
                continue
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def required_metal_fragment_builtin_parameters(
        self, func, reserved_names=None, explicit_stage_builtins=None
    ):
        used_names = self.used_metal_fragment_builtin_names(getattr(func, "body", []))
        reserved_names = set(reserved_names or ())
        explicit_stage_builtins = explicit_stage_builtins or {}
        required_parameters = []
        for (
            builtin_name,
            param_type,
            attribute,
        ) in self.metal_fragment_builtin_parameter_specs():
            if builtin_name not in used_names or attribute in explicit_stage_builtins:
                continue
            name = self.unique_metal_generated_name("_crossglFragCoord", reserved_names)
            reserved_names.add(name)
            required_parameters.append((builtin_name, name, param_type, attribute))
        return required_parameters

    def used_metal_fragment_builtin_names(self, body):
        builtin_names = {"gl_FragCoord"}
        used_names = set()
        for node in self.iter_ast_nodes(body):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name:
                continue
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def validate_metal_fragment_invocation_density(self, func, shader_type):
        if shader_type != "fragment":
            return
        for reachable_func in self.metal_reachable_functions(func):
            if self.function_uses_metal_fragment_invocation_density(reachable_func):
                raise UnsupportedMetalFeatureError(
                    "gl_FragSizeEXT",
                    (
                        "Metal target does not expose a fragment invocation density "
                        "or fragment-size input equivalent for "
                        "GL_EXT_fragment_invocation_density / gl_FragSizeEXT; keep "
                        "this shader on a target with fragment density builtins or "
                        "specialize the density path before Metal generation."
                    ),
                    missing_capabilities=FRAGMENT_INVOCATION_DENSITY_CAPABILITIES,
                )

    def metal_reachable_functions(self, root_func):
        reachable = []
        pending = [root_func]
        visited = set()
        while pending:
            func = pending.pop(0)
            if func is None or id(func) in visited:
                continue
            visited.add(id(func))
            reachable.append(func)
            for called_name in sorted(self.called_user_function_names(func)):
                called_func = (self.functions_by_name or {}).get(called_name)
                if called_func is not None and id(called_func) not in visited:
                    pending.append(called_func)
        return reachable

    def function_uses_metal_fragment_invocation_density(self, func):
        for node in self.iter_ast_nodes(getattr(func, "body", []) or []):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name and class_name != "VariableNode":
                continue
            name = getattr(node, "name", "")
            if name.split(".", 1)[0] == "gl_FragSizeEXT":
                return True
        return False

    def validate_graphics_builtin_parameter_types(self, parameters, stage_name):
        expected_types = {
            "vertex_id": "uint",
            "instance_id": "uint",
            "primitive_id": "uint",
            "is_front_facing": "bool",
            "position": "float4",
            "point_coord": "float2",
            "sample_id": "uint",
            "sample_mask": "uint",
        }
        for parameter in parameters or []:
            if self.is_graphics_stage_output_parameter(parameter, stage_name):
                continue
            semantic = self.semantic_from_node(parameter)
            if stage_name == "fragment" and str(semantic) == "gl_SampleMask":
                raise ValueError(
                    "Metal fragment stage gl_SampleMask parameter is output-only; "
                    "use gl_SampleMaskIn or sample_mask for coverage-mask input"
                )
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = expected_types.get(metal_semantic)
            if expected_type is None:
                continue
            actual_type = self.map_type(self.parameter_raw_type(parameter))
            if self.metal_graphics_builtin_parameter_lowering(
                self.parameter_raw_type(parameter),
                actual_type,
                parameter,
                stage_name,
            ):
                continue
            if actual_type != expected_type:
                name = getattr(parameter, "name", "<anonymous>")
                raise ValueError(
                    f"Metal {stage_name} semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires parameter '{name}' to "
                    f"have type {expected_type}, got {actual_type}"
                )

    def is_graphics_stage_output_parameter(self, parameter, stage_name):
        if stage_name not in {"vertex", "fragment"}:
            return False
        if self.is_metal_mesh_output_parameter(stage_name, parameter):
            return False
        if not (self.parameter_qualifier_names(parameter) & {"out", "inout"}):
            return False
        return not self.is_resource_parameter_type(self.parameter_raw_type(parameter))

    def is_spirv_stage_interface_layout(self, node):
        return (
            getattr(node, "layout_type", None) in {"IN", "OUT"}
            and getattr(node, "data_type", None) is not None
            and getattr(node, "variable_name", None) is not None
        )

    def spirv_stage_input_layouts(self, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return []
        return [
            node
            for node in getattr(self, "metal_spirv_interface_variables", []) or []
            if getattr(node, "layout_type", None) == "IN"
        ]

    def spirv_stage_output_layouts(self, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return []
        return [
            node
            for node in getattr(self, "metal_spirv_interface_variables", []) or []
            if getattr(node, "layout_type", None) == "OUT"
            and self.spirv_stage_output_semantic(node) is not None
        ]

    def spirv_stage_layout_qualifier_value(self, node, qualifier_name):
        for name, value in getattr(node, "qualifiers", []) or []:
            if str(name).lower() == qualifier_name:
                return value
        return None

    def spirv_stage_input_semantic(self, node):
        location = self.spirv_stage_layout_qualifier_value(node, "location")
        if location is not None:
            return f"attribute({location})"
        builtin = self.spirv_stage_layout_qualifier_value(node, "builtin")
        return self.spirv_builtin_semantic(builtin)

    def spirv_stage_output_semantic(self, node):
        builtin = self.spirv_stage_layout_qualifier_value(node, "builtin")
        if builtin is not None:
            return self.spirv_builtin_semantic(builtin)
        location = self.spirv_stage_layout_qualifier_value(node, "location")
        if location is not None:
            return f"color({location})"
        return None

    def spirv_stage_input_attribute(self, node):
        semantic = self.spirv_stage_input_semantic(node)
        return self.map_semantic(semantic) if semantic is not None else None

    def spirv_stage_output_attribute(self, node):
        semantic = self.spirv_stage_output_semantic(node)
        return self.map_semantic(semantic) if semantic is not None else ""

    def spirv_builtin_semantic(self, builtin):
        if builtin is None:
            return None
        return {
            "Position": "gl_Position",
            "PointSize": "gl_PointSize",
            "ClipDistance": "gl_ClipDistance",
            "FragCoord": "gl_FragCoord",
            "FragDepth": "gl_FragDepth",
        }.get(str(builtin), str(builtin))

    def metal_graphics_builtin_parameter_lowering(
        self,
        raw_param_type,
        mapped_param_type,
        parameter,
        stage_name,
        reserved_names=None,
    ):
        if stage_name != "vertex":
            return None
        metal_semantic = self.canonical_metal_semantic(
            self.semantic_from_node(parameter)
        )
        if metal_semantic not in {"vertex_id", "instance_id"}:
            return None
        if mapped_param_type != "int":
            return None

        base_name = {
            "vertex_id": "_crossglVertexID",
            "instance_id": "_crossglInstanceID",
        }[metal_semantic]
        return {
            "type": "uint",
            "name": self.unique_metal_generated_name(base_name, reserved_names or ()),
            "attribute": metal_semantic,
            "source_type": raw_param_type,
        }

    def is_plain_metal_vertex_stage_input_parameter(
        self, raw_param_type, mapped_param_type, parameter, shader_type
    ):
        if shader_type != "vertex":
            return False
        if self.is_graphics_stage_output_parameter(parameter, shader_type):
            return False
        if (
            self.semantic_from_node(parameter) is not None
            and self.metal_vertex_input_location_semantic(parameter) is None
        ):
            return False
        if self.is_resource_parameter_type(raw_param_type):
            return False
        if isinstance(raw_param_type, (PointerType, ReferenceType)):
            return False
        if self.metal_parameter_user_struct_type(parameter) is not None:
            return False

        mapped_type = self.type_name_string(mapped_param_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return False
        if self.metal_matrix_dimensions(base_type) is not None:
            return False
        if base_type in self.structs_by_name:
            return False
        return self.is_metal_scalar_or_vector_type_name(base_type)

    def is_plain_metal_fragment_stage_input_parameter(
        self, raw_param_type, mapped_param_type, parameter, shader_type
    ):
        if shader_type != "fragment":
            return False
        if self.is_graphics_stage_output_parameter(parameter, shader_type):
            return False
        semantic = self.semantic_from_node(parameter)
        if not self.is_metal_user_stage_io_semantic(semantic):
            return False
        if self.is_resource_parameter_type(raw_param_type):
            return False
        if isinstance(raw_param_type, (PointerType, ReferenceType)):
            return False
        if self.metal_parameter_user_struct_type(parameter) is not None:
            return False

        mapped_type = self.type_name_string(mapped_param_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return False
        if self.metal_matrix_dimensions(base_type) is not None:
            return False
        if base_type in self.structs_by_name:
            return False
        return self.is_metal_scalar_or_vector_type_name(base_type)

    def metal_fragment_input_semantic_attribute(self, semantic):
        if self.metal_attribute_index_from_semantic(semantic) is not None:
            return self.map_semantic(semantic)
        hlsl_attribute = self.metal_hlsl_attribute_semantic(semantic)
        if hlsl_attribute is not None:
            return f" [[{hlsl_attribute}]]"
        hlsl_color = self.metal_hlsl_color_semantic(semantic)
        if hlsl_color is not None:
            return f" [[user({hlsl_color})]]"
        if str(semantic).lower() == "color":
            return " [[user(Color0)]]"
        return None

    def is_metal_scalar_or_vector_type_name(self, type_name):
        type_name = str(type_name).strip()
        return (
            re.fullmatch(
                r"(?:packed_)?(?:half|float|double|bool|char|uchar|short|ushort|"
                r"int|uint|long|ulong)(?:[234])?",
                type_name,
            )
            is not None
        )

    def unique_vertex_stage_input_struct_name(self, function_name):
        base_name = f"{function_name}_Input"
        if base_name[:1].isdigit():
            base_name = f"Generated_{base_name}"
        used_names = set(self.structs_by_name)
        used_names.update(self.generated_return_wrapper_struct_names)
        used_names.update(getattr(self, "metal_generated_struct_names", set()))
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self.generated_return_wrapper_struct_names.add(candidate)
        return candidate

    def generate_metal_vertex_stage_input_parameter_struct(
        self, struct_name, parameters
    ):
        code = f"struct {struct_name} {{\n"
        used_attributes = {
            attribute_index
            for parameter in parameters
            for attribute_index in [
                self.metal_attribute_index_from_attribute(parameter.get("attribute"))
            ]
            if attribute_index is not None
        }
        next_attribute = 0
        for parameter in parameters:
            attribute = parameter.get("attribute")
            attribute_index = self.metal_attribute_index_from_attribute(attribute)
            if attribute_index is None and (
                attribute is None
                or not attribute.strip()
                or (
                    self.is_metal_user_stage_io_semantic(parameter.get("semantic"))
                    and not parameter.get("preserve_user_attribute")
                )
            ):
                while next_attribute in used_attributes:
                    next_attribute += 1
                attribute = f" [[attribute({next_attribute})]]"
                used_attributes.add(next_attribute)
                next_attribute += 1
            if "declaration" in parameter:
                code += f"    {parameter['declaration']}{attribute or ''};\n"
            else:
                field_decl = format_c_style_array_declaration(
                    parameter["mapped_type"], parameter["name"]
                )
                code += f"    {field_decl}{attribute or ''};\n"
        code += "};\n\n"
        return code

    def generate_metal_vertex_stage_input_alias_declarations(
        self, input_parameter_name, parameters
    ):
        declarations = []
        for parameter in parameters:
            declaration = parameter.get("declaration")
            if declaration is None:
                declaration = format_c_style_array_declaration(
                    parameter["mapped_type"],
                    self.metal_local_identifier_name(parameter["name"]),
                )
            declarations.append(
                f"{declaration} = {input_parameter_name}.{parameter['name']};"
            )
        return declarations

    def metal_stage_output_parameter_attribute(self, parameter, stage_name):
        semantic = self.semantic_from_node(parameter)
        if semantic is None and stage_name == "fragment":
            semantic = "gl_FragColor"
        return self.map_semantic(semantic)

    def generate_metal_stage_output_parameter_struct(self, struct_name, parameters):
        code = f"struct {struct_name} {{\n"
        for parameter in parameters:
            field_decl = format_c_style_array_declaration(
                parameter["mapped_type"], parameter["name"]
            )
            code += f"    {field_decl}{parameter['attribute']};\n"
        code += "};\n\n"
        return code

    def generate_metal_stage_output_parameter_locals(self, parameters):
        code = ""
        for parameter in parameters:
            default_value = self.metal_default_value_expression(parameter["raw_type"])
            declaration = format_c_style_array_declaration(
                parameter["mapped_type"],
                self.metal_local_identifier_name(parameter["name"]),
            )
            code += f"    {declaration} = {default_value};\n"
        return code

    def generate_metal_stage_output_parameter_return(
        self, struct_name, parameters, reserved_names
    ):
        if not parameters or not struct_name:
            return ""
        result_name = self.unique_metal_generated_name("_crossglOutput", reserved_names)
        code = f"    {struct_name} {result_name};\n"
        for parameter in parameters:
            local_name = self.metal_local_identifier_name(parameter["name"])
            code += f"    {result_name}.{parameter['name']} = {local_name};\n"
        code += f"    return {result_name};\n"
        return code

    def validate_struct_member_semantic_types(self, struct_node):
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = self.struct_member_builtin_semantic_type(metal_semantic)
            if expected_type is None:
                continue
            if expected_type == "invalid":
                struct_name = getattr(struct_node, "name", "<anonymous>")
                member_name = getattr(member, "name", "<anonymous>")
                raise ValueError(
                    f"Metal struct '{struct_name}' semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and is not supported"
                )
            actual_type = self.map_type(self.struct_member_raw_type(member))
            if actual_type != expected_type:
                struct_name = getattr(struct_node, "name", "<anonymous>")
                member_name = getattr(member, "name", "<anonymous>")
                raise ValueError(
                    f"Metal struct '{struct_name}' semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires member '{member_name}' to "
                    f"have type {expected_type}, got {actual_type}"
                )

    def struct_member_builtin_semantic_type(self, metal_semantic):
        fixed_types = {
            "vertex_id": "uint",
            "instance_id": "uint",
            "primitive_id": "uint",
            "is_front_facing": "bool",
            "position": "float4",
            "point_coord": "float2",
            "point_size": "float",
            "depth(any)": "float",
            "sample_id": "uint",
            "sample_mask": "uint",
        }
        if metal_semantic == "stencil_ref":
            return "invalid"
        if metal_semantic is None:
            return None
        if metal_semantic.startswith("color("):
            return "float4"
        return fixed_types.get(metal_semantic)

    def struct_member_raw_type(self, member):
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            raw_type = self.type_name_string(element_type) or "float"
            if getattr(member, "size", None) is None:
                return f"{raw_type}[]"
            return f"{raw_type}[{member.size}]"
        if hasattr(member, "member_type"):
            return self.type_name_string(member.member_type)
        if hasattr(member, "vtype"):
            return self.type_name_string(member.vtype)
        return "float"

    def function_return_semantic_wrapper(
        self, func, raw_return_type, mapped_return_type, stage_name, function_name
    ):
        semantic = self.function_return_semantic(func)
        metal_semantic = self.canonical_metal_semantic(semantic)
        if not self.function_return_semantic_requires_wrapper(semantic, metal_semantic):
            return None
        expected_type = self.function_return_builtin_semantic_type(
            semantic, metal_semantic
        )
        if expected_type is None or expected_type == "invalid":
            return None
        if mapped_return_type != expected_type:
            return None
        struct_name = self.unique_return_wrapper_struct_name(function_name)
        return {
            "struct_name": struct_name,
            "field_type": mapped_return_type,
            "field_name": self.return_wrapper_field_name(semantic, metal_semantic),
            "source_type": raw_return_type,
            "semantic_attr": self.map_semantic(semantic),
        }

    def function_return_semantic_requires_wrapper(self, semantic, metal_semantic):
        if semantic is None:
            return False
        semantic = str(semantic)
        if semantic == "gl_FragDepth":
            return True
        if semantic == "gl_FragStencilRefEXT":
            return True
        if metal_semantic == "sample_mask":
            return True
        return bool(
            metal_semantic
            and metal_semantic.startswith("color(")
            and metal_semantic != "color(0)"
        )

    def unique_return_wrapper_struct_name(self, function_name):
        base_name = f"{function_name}_Return"
        if base_name[:1].isdigit():
            base_name = f"Generated_{base_name}"
        used_names = set(self.structs_by_name)
        used_names.update(self.generated_return_wrapper_struct_names)
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self.generated_return_wrapper_struct_names.add(candidate)
        return candidate

    def return_wrapper_field_name(self, semantic, metal_semantic):
        semantic = str(semantic)
        if semantic == "gl_FragDepth":
            return "depth"
        if semantic == "gl_PointSize":
            return "pointSize"
        if metal_semantic == "sample_mask":
            return "sampleMask"
        if metal_semantic and metal_semantic.startswith("color("):
            return "color"
        return "value"

    def generate_return_wrapper_struct(self, wrapper):
        return (
            f"struct {wrapper['struct_name']} {{\n"
            f"    {wrapper['field_type']} {wrapper['field_name']}"
            f"{wrapper['semantic_attr']};\n"
            "};\n\n"
        )

    def validate_function_return_semantic_type(self, func, raw_return_type, stage_name):
        semantic = self.function_return_semantic(func)
        if semantic is not None and self.map_type(raw_return_type) == "void":
            function_name = getattr(func, "name", "<anonymous>")
            raise ValueError(
                f"Metal {stage_name} function '{function_name}' cannot use "
                f"return semantic '{semantic}' with void return type"
            )

        metal_semantic = self.canonical_metal_semantic(semantic)
        self.validate_function_return_semantic_stage(
            stage_name, semantic, metal_semantic
        )
        expected_type = self.function_return_builtin_semantic_type(
            semantic, metal_semantic
        )
        if expected_type is None:
            return
        if expected_type == "invalid":
            raise ValueError(
                f"Metal {stage_name} semantic '{semantic}' cannot be used as a "
                "function return semantic"
            )
        actual_type = self.map_type(raw_return_type)
        if actual_type != expected_type:
            function_name = getattr(func, "name", "<anonymous>")
            raise ValueError(
                f"Metal {stage_name} return semantic '{semantic}' maps to "
                f"'{metal_semantic}' and requires function '{function_name}' "
                f"to return {expected_type}, got {actual_type}"
            )

    def validate_function_return_semantic_stage(
        self, stage_name, semantic, metal_semantic
    ):
        if semantic is None:
            return

        semantic_text = str(semantic)
        is_fragment_output = semantic_text in {
            "gl_FragDepth",
            "gl_FragStencilRefEXT",
        } or bool(metal_semantic and metal_semantic.startswith("color("))
        is_fragment_output = is_fragment_output or metal_semantic in {
            "sample_mask",
            "stencil_ref",
        }
        is_vertex_output = semantic_text == "gl_Position"

        if stage_name == "vertex" and is_fragment_output:
            raise ValueError(
                f"Metal vertex stage function return cannot use fragment output "
                f"semantic '{semantic}'"
            )
        if stage_name == "fragment" and is_vertex_output:
            raise ValueError(
                f"Metal fragment stage function return cannot use vertex output "
                f"semantic '{semantic}'"
            )

    def function_return_builtin_semantic_type(self, semantic, metal_semantic):
        if semantic is None:
            return None
        semantic = str(semantic)
        invalid_return_semantics = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_IsFrontFace",
            "gl_PointSize",
            "gl_FragCoord",
            "gl_FrontFacing",
            "gl_PointCoord",
            "gl_SampleID",
            "gl_SampleMaskIn",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        if semantic in invalid_return_semantics:
            return "invalid"
        output_types = {
            "gl_Position": "float4",
            "gl_FragDepth": "float",
            "gl_SampleMask": "uint",
        }
        if semantic == "gl_FragStencilRefEXT" or metal_semantic == "stencil_ref":
            return "invalid"
        if semantic in output_types:
            return output_types[semantic]
        if metal_semantic and metal_semantic.startswith("color("):
            return "float4"
        if metal_semantic == "sample_mask":
            return "uint"
        return None

    def canonical_metal_semantic(self, semantic):
        if semantic is None:
            return None
        mapped_semantic = self.semantic_map.get(str(semantic), str(semantic))
        if mapped_semantic.startswith("[[") and mapped_semantic.endswith("]]"):
            return mapped_semantic[2:-2]
        return mapped_semantic

    def append_global_resource_parameters(
        self, params_str, func_name=None, func=None, filter_writable_textures=False
    ):
        resource_params = []
        dependencies = None
        entry_point_dependencies = (
            self.entry_point_global_resource_dependencies(func)
            if func is not None
            else None
        )
        texture_dependencies = (
            entry_point_dependencies if filter_writable_textures else None
        )
        sampler_dependencies = (
            entry_point_dependencies
            if func is not None
            else (
                self.function_global_resource_dependencies.get(func_name, set())
                if func_name
                else None
            )
        )
        cbuffer_dependencies = None
        if self.cbuffer_variables:
            for cbuffer in self.cbuffer_variables:
                cbuffer_name = getattr(cbuffer, "name", None)
                if (
                    cbuffer_dependencies is not None
                    and cbuffer_name not in cbuffer_dependencies
                ):
                    continue
                binding = self.cbuffer_binding_indices.get(id(cbuffer), 0)
                parameter_name = self.cbuffer_parameter_name(cbuffer)
                resource_params.append(
                    f"constant {cbuffer.name}& {parameter_name} [[buffer({binding})]]"
                )
        if self.metal_buffer_resource_variables:
            for (
                buffer_variable,
                i,
                buffer_type,
                array_size,
                address_space,
            ) in self.metal_buffer_resource_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_metal_buffer_resource_parameter(
                    buffer_type,
                    buffer_name,
                    array_size,
                    address_space,
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.texture_variables:
            for (
                texture_variable,
                i,
                texture_type,
                array_size,
            ) in self.texture_variables:
                texture_name = getattr(texture_variable, "name", None)
                if (
                    texture_dependencies is not None
                    and texture_name not in texture_dependencies
                    and self.metal_texture_parameter_type_is_writable(texture_type)
                ):
                    continue
                declaration = self.format_resource_parameter(
                    texture_type, texture_name, array_size
                )
                resource_params.append(f"{declaration} [[texture({i})]]")
        if self.acceleration_structure_variables:
            for (
                acceleration_structure_variable,
                i,
                acceleration_structure_type,
                array_size,
            ) in self.acceleration_structure_variables:
                acceleration_structure_name = getattr(
                    acceleration_structure_variable, "name", None
                )
                if (
                    dependencies is not None
                    and acceleration_structure_name not in dependencies
                ):
                    continue
                declaration = self.format_resource_parameter(
                    acceleration_structure_type,
                    acceleration_structure_name,
                    array_size,
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.visible_function_table_variables:
            for (
                visible_function_table_variable,
                i,
                visible_function_table_type,
                array_size,
            ) in self.visible_function_table_variables:
                visible_function_table_name = getattr(
                    visible_function_table_variable, "name", None
                )
                if (
                    dependencies is not None
                    and visible_function_table_name not in dependencies
                ):
                    continue
                declaration = self.format_visible_function_table_parameter(
                    visible_function_table_type, visible_function_table_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.intersection_function_table_variables:
            for (
                intersection_function_table_variable,
                i,
                intersection_function_table_type,
                array_size,
            ) in self.intersection_function_table_variables:
                intersection_function_table_name = getattr(
                    intersection_function_table_variable, "name", None
                )
                if (
                    dependencies is not None
                    and intersection_function_table_name not in dependencies
                ):
                    continue
                declaration = self.format_intersection_function_table_parameter(
                    intersection_function_table_type,
                    intersection_function_table_name,
                    array_size,
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.structured_buffer_variables:
            for (
                buffer_variable,
                i,
                buffer_type,
                array_size,
            ) in self.structured_buffer_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_structured_buffer_parameter(
                    buffer_type, buffer_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.structured_buffer_length_variables:
            for (
                buffer_variable,
                i,
                _buffer_type,
                array_size,
            ) in self.structured_buffer_length_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_structured_buffer_length_parameter(
                    buffer_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.structured_buffer_counter_variables:
            for (
                buffer_variable,
                i,
                _buffer_type,
                array_size,
            ) in self.structured_buffer_counter_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_structured_buffer_counter_parameter(
                    buffer_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.glsl_buffer_block_variables:
            for (
                buffer_variable,
                i,
                block,
                array_size,
            ) in self.glsl_buffer_block_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_glsl_buffer_block_parameter(
                    block, buffer_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.sampler_variables:
            for sampler_variable, i, array_size in self.sampler_variables:
                sampler_name = getattr(sampler_variable, "name", None)
                if (
                    sampler_dependencies is not None
                    and sampler_name not in sampler_dependencies
                ):
                    continue
                declaration = self.format_resource_parameter(
                    "sampler", sampler_name, array_size
                )
                resource_params.append(f"{declaration} [[sampler({i})]]")
        if not resource_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(resource_params)}"
        return ", ".join(resource_params)

    def entry_point_global_resource_dependencies(self, func):
        dependencies = set(self.direct_global_resource_dependencies(func))
        for called_name in self.called_user_function_names(func):
            dependencies.update(
                self.function_global_resource_dependencies.get(called_name, set())
            )
        return dependencies

    def entry_point_cbuffer_dependencies(self, func):
        dependencies = set(self.direct_cbuffer_dependencies(func))
        for called_name in self.called_user_function_names(func):
            dependencies.update(
                self.function_cbuffer_dependencies.get(called_name, set())
            )
        return dependencies

    def metal_texture_parameter_type_is_writable(self, texture_type):
        texture_type = str(texture_type or "")
        return "access::write" in texture_type or "access::read_write" in texture_type

    def global_resource_parameter_names(self):
        names = set()
        for texture_variable, _, _, _ in self.texture_variables:
            if getattr(texture_variable, "name", None):
                names.add(texture_variable.name)
        for (
            acceleration_structure_variable,
            _,
            _,
            _,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None):
                names.add(acceleration_structure_variable.name)
        for (
            visible_function_table_variable,
            _,
            _,
            _,
        ) in self.visible_function_table_variables:
            if getattr(visible_function_table_variable, "name", None):
                names.add(visible_function_table_variable.name)
        for (
            intersection_function_table_variable,
            _,
            _,
            _,
        ) in self.intersection_function_table_variables:
            if getattr(intersection_function_table_variable, "name", None):
                names.add(intersection_function_table_variable.name)
        for buffer_variable, _, _, _ in self.structured_buffer_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for buffer_variable, _, _, _, _ in self.metal_buffer_resource_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for buffer_variable, _, _, _ in self.structured_buffer_length_variables:
            if getattr(buffer_variable, "name", None):
                names.add(
                    self.structured_buffer_length_parameter_name(buffer_variable.name)
                )
        for buffer_variable, _, _, _ in self.structured_buffer_counter_variables:
            if getattr(buffer_variable, "name", None):
                names.add(
                    self.structured_buffer_counter_parameter_name(buffer_variable.name)
                )
        for buffer_variable, _, _, _ in self.glsl_buffer_block_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for sampler_variable, _, _ in self.sampler_variables:
            if getattr(sampler_variable, "name", None):
                names.add(sampler_variable.name)
        return names

    def append_required_cbuffer_parameters(self, params_str, func_name):
        cbuffer_params = []
        for cbuffer in self.required_function_cbuffers(func_name):
            parameter_name = self.cbuffer_parameter_name(cbuffer)
            cbuffer_params.append(f"constant {cbuffer.name}& {parameter_name}")
        if not cbuffer_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(cbuffer_params)}"
        return ", ".join(cbuffer_params)

    def append_required_stage_parameter_parameters(self, params_str, func_name):
        stage_params = []
        for parameter in self.required_function_stage_parameters(func_name):
            raw_type = self.parameter_raw_type(parameter)
            mapped_type = self.map_type(raw_type)
            stage_params.append(
                format_c_style_array_declaration(mapped_type, parameter.name)
            )
        if not stage_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(stage_params)}"
        return ", ".join(stage_params)

    def append_required_stage_output_parameter(self, params_str, func_name):
        output_type = self.required_function_stage_output_type(func_name)
        if output_type is None:
            return params_str
        declaration = f"thread {self.map_type(output_type)}& output"
        if params_str:
            return f"{params_str}, {declaration}"
        return declaration

    def append_required_global_resource_parameters(self, params_str, func_name):
        resource_params = []
        for (
            texture_variable,
            texture_type,
            array_size,
        ) in self.required_function_textures(func_name):
            texture_name = getattr(texture_variable, "name", None)
            if texture_name:
                resource_params.append(
                    self.format_resource_parameter(
                        texture_type, texture_name, array_size
                    )
                )
        for (
            acceleration_structure_variable,
            acceleration_structure_type,
            array_size,
        ) in self.required_function_acceleration_structures(func_name):
            acceleration_structure_name = getattr(
                acceleration_structure_variable, "name", None
            )
            if acceleration_structure_name:
                resource_params.append(
                    self.format_resource_parameter(
                        acceleration_structure_type,
                        acceleration_structure_name,
                        array_size,
                    )
                )
        for (
            visible_function_table_variable,
            visible_function_table_type,
            array_size,
        ) in self.required_function_visible_function_tables(func_name):
            visible_function_table_name = getattr(
                visible_function_table_variable, "name", None
            )
            if visible_function_table_name:
                resource_params.append(
                    self.format_visible_function_table_parameter(
                        visible_function_table_type,
                        visible_function_table_name,
                        array_size,
                    )
                )
        for (
            intersection_function_table_variable,
            intersection_function_table_type,
            array_size,
        ) in self.required_function_intersection_function_tables(func_name):
            intersection_function_table_name = getattr(
                intersection_function_table_variable, "name", None
            )
            if intersection_function_table_name:
                resource_params.append(
                    self.format_intersection_function_table_parameter(
                        intersection_function_table_type,
                        intersection_function_table_name,
                        array_size,
                    )
                )
        for (
            buffer_variable,
            buffer_type,
            array_size,
            address_space,
        ) in self.required_function_metal_buffer_resources(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_metal_buffer_resource_parameter(
                        buffer_type, buffer_name, array_size, address_space
                    )
                )
        for (
            buffer_variable,
            buffer_type,
            array_size,
        ) in self.required_function_structured_buffers(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_structured_buffer_parameter(
                        buffer_type, buffer_name, array_size
                    )
                )
                if self.structured_buffer_requires_length(buffer_name):
                    resource_params.append(
                        self.format_structured_buffer_length_parameter(
                            buffer_name, array_size
                        )
                    )
                if self.structured_buffer_requires_counter(buffer_type):
                    resource_params.append(
                        self.format_structured_buffer_counter_parameter(
                            buffer_name, array_size
                        )
                    )
        for (
            buffer_variable,
            block,
            array_size,
        ) in self.required_function_glsl_buffer_blocks(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_glsl_buffer_block_parameter(
                        block, buffer_name, array_size
                    )
                )
        for sampler_variable, array_size in self.required_function_samplers(func_name):
            sampler_name = getattr(sampler_variable, "name", None)
            if sampler_name:
                resource_params.append(
                    self.format_resource_parameter("sampler", sampler_name, array_size)
                )
        if not resource_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(resource_params)}"
        return ", ".join(resource_params)

    def cbuffer_parameter_name(self, cbuffer):
        parameter_name = self.cbuffer_parameter_names.get(id(cbuffer))
        if parameter_name:
            return parameter_name
        return self.default_cbuffer_parameter_name(cbuffer)

    def default_cbuffer_parameter_name(self, cbuffer):
        name = getattr(cbuffer, "name", "constants")
        if not name:
            return "constants"
        return name[:1].lower() + name[1:]

    def collect_cbuffer_parameter_names(self, cbuffers, reserved_names=None):
        parameter_names = {}
        used_names = set(reserved_names or [])
        for cbuffer in cbuffers:
            base_name = self.default_cbuffer_parameter_name(cbuffer)
            parameter_name = base_name
            suffix = 1
            while parameter_name in used_names:
                parameter_name = f"{base_name}{suffix}"
                suffix += 1
            used_names.add(parameter_name)
            parameter_names[id(cbuffer)] = parameter_name
        return parameter_names

    def collect_cbuffer_member_references(self, cbuffers):
        references = {}
        ambiguous_members = collect_duplicate_cbuffer_member_names(cbuffers)
        for cbuffer in cbuffers:
            parameter_name = self.cbuffer_parameter_name(cbuffer)
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name or member_name in ambiguous_members:
                    continue
                references[member_name] = f"{parameter_name}.{member_name}"
        self.ambiguous_cbuffer_members = ambiguous_members
        return references

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as Metal source."""
        indent_str = "    " * indent
        if isinstance(stmt, (VariableNode, BackendVariableNode)):
            var_type = self.local_variable_declared_type(stmt)
            texture_alias_type = self.local_variable_texture_alias_resource_type(
                stmt, var_type
            )
            if texture_alias_type is not None:
                initial_raw_type = self.texture_argument_raw_resource_type(
                    getattr(stmt, "initial_value", None)
                )
                var_type = self.local_variable_texture_alias_declaration_type(
                    stmt, texture_alias_type
                )
                self.current_texture_parameters[stmt.name] = texture_alias_type
                if initial_raw_type is not None:
                    self.current_texture_parameter_raw_types[stmt.name] = (
                        self.resource_base_type(initial_raw_type)
                    )
                self.record_local_texture_alias_sampler_source(stmt)
                self.record_local_image_alias_metadata(stmt)
            sampler_alias_type = self.local_variable_sampler_alias_resource_type(
                stmt, var_type
            )
            if sampler_alias_type is not None:
                var_type = sampler_alias_type
                self.current_sampler_parameters.add(stmt.name)
                _sampler_type, sampler_array_size = self.metal_array_type_parts(
                    sampler_alias_type
                )
                if sampler_array_size is not None:
                    self.current_sampler_parameter_array_sizes[stmt.name] = (
                        sampler_array_size
                    )
            acceleration_structure_alias_type = (
                self.local_variable_acceleration_structure_alias_resource_type(
                    stmt, var_type
                )
            )
            if acceleration_structure_alias_type is not None:
                var_type = acceleration_structure_alias_type
            initial_value = getattr(stmt, "initial_value", None)
            acceleration_structure_array_alias = (
                self.unsupported_metal_acceleration_structure_array_alias_source(
                    initial_value
                )
            )
            if (
                acceleration_structure_array_alias is not None
                and self.is_acceleration_structure_type(var_type)
            ):
                self.local_variable_types[stmt.name] = var_type
                self.unsupported_metal_acceleration_structure_array_variables[
                    stmt.name
                ] = self.unsupported_metal_acceleration_structure_array_variables[
                    acceleration_structure_array_alias
                ]
                diagnostic = self.unsupported_metal_acceleration_structure_array_alias_diagnostic(
                    stmt.name, acceleration_structure_array_alias
                )
                return f"{indent_str}{diagnostic}"
            ray_function_table_array_alias = (
                self.unsupported_metal_ray_function_table_array_alias_source(
                    initial_value
                )
            )
            if ray_function_table_array_alias is not None:
                self.local_variable_types[stmt.name] = var_type
                self.unsupported_metal_ray_function_table_array_variables[stmt.name] = (
                    self.unsupported_metal_ray_function_table_array_variables[
                        ray_function_table_array_alias
                    ]
                )
                diagnostic = (
                    self.unsupported_metal_ray_function_table_array_alias_diagnostic(
                        stmt.name, ray_function_table_array_alias
                    )
                )
                return f"{indent_str}{diagnostic}"
            self.local_variable_types[stmt.name] = var_type
            self.current_address_space_variables[stmt.name] = (
                self.local_variable_address_space(stmt)
            )
            self.record_readonly_metal_mesh_payload_local_alias(stmt)
            self.record_readonly_metal_local_alias(stmt)
            is_atomic_local = self.is_metal_atomic_value_type(var_type)
            if self.is_unsupported_glsl_buffer_block_struct_type(var_type):
                self.current_unsupported_glsl_buffer_block_local_variables.add(
                    stmt.name
                )
                return (
                    f"{indent_str}"
                    f"{self.unsupported_glsl_buffer_block_local_variable_placeholder('Metal', var_type, stmt.name)};\n"
                )

            local_name = self.metal_local_identifier_name(stmt.name)
            declaration = format_c_style_array_declaration(
                self.map_type(var_type), local_name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            declaration = self.maybe_format_unused_local_declaration(
                declaration, stmt.name
            )
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    local_name,
                    var_type,
                    indent,
                    "Metal",
                )
                return code
            if initial_value is not None:
                address_space_conflict = (
                    self.local_variable_address_space_conflict_diagnostic(stmt)
                )
                if address_space_conflict is not None:
                    conflict_declaration = (
                        self.local_variable_address_space_conflict_declaration(
                            stmt, declaration
                        )
                    )
                    return (
                        f"{indent_str}{address_space_conflict}\n"
                        f"{indent_str}{conflict_declaration};\n"
                    )
                init_expr = self.generate_expression_with_expected(
                    initial_value, var_type
                )
                if self.local_pointer_initializer_needs_address(stmt, initial_value):
                    init_expr = f"&{init_expr}"
                address_space_mismatch = (
                    self.local_variable_address_space_mismatch_diagnostic(stmt)
                )
                if is_atomic_local:
                    array_initializer = (
                        self.generate_metal_atomic_array_initializer_stores(
                            local_name, var_type, initial_value, indent_str
                        )
                    )
                    if array_initializer is not None:
                        return f"{indent_str}{declaration};\n{array_initializer}"
                    return (
                        f"{indent_str}{declaration};\n"
                        f"{indent_str}atomic_store_explicit(&{local_name}, {init_expr}, "
                        "memory_order_relaxed);\n"
                    )
                diagnostic_prefix = (
                    f"{indent_str}{address_space_mismatch}\n"
                    if address_space_mismatch is not None
                    else ""
                )
                return f"{diagnostic_prefix}{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, ArrayNode):
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)
            local_name = self.metal_local_identifier_name(stmt.name)

            if size is None:
                # Dynamic arrays in Metal need a size, use a large enough buffer
                declaration = self.maybe_format_unused_local_declaration(
                    f"device array<{element_type}, 1024> {local_name}",
                    stmt.name,
                )
                return f"{indent_str}{declaration};\n"
            else:
                declaration = self.maybe_format_unused_local_declaration(
                    f"array<{element_type}, {size}> {local_name}",
                    stmt.name,
                )
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, (AssignmentNode, BackendAssignmentNode)):
            return self.generate_statement_code(self.generate_assignment(stmt), indent)
        elif isinstance(stmt, BlockNode):
            return self.generate_block(stmt, indent)
        elif isinstance(stmt, BreakNode):
            return f"{indent_str}break;\n"
        elif isinstance(stmt, ContinueNode):
            return f"{indent_str}continue;\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)
        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)
        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)
        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)
        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)
        elif isinstance(stmt, (ReturnNode, BackendReturnNode)):
            if getattr(stmt, "value", None) is None:
                if getattr(self, "current_metal_stage_output_return_active", False):
                    return ""
                return f"{indent_str}return;\n"
            if isinstance(stmt.value, list):
                code = ""
                for i, return_stmt in enumerate(stmt.value):
                    code += f"{self.generate_expression(return_stmt)}"
                    if i < len(stmt.value) - 1:
                        code += ", "
                return f"{indent_str}return {code};\n"
            else:
                return_wrapper = self.current_function_return_wrapper
                if return_wrapper is not None:
                    value = self.generate_expression_with_expected(
                        stmt.value, return_wrapper["source_type"]
                    )
                    return (
                        f"{indent_str}return {return_wrapper['struct_name']}"
                        f"{{{value}}};\n"
                    )
                value = self.generate_expression_with_expected(
                    stmt.value, self.current_function_return_type
                )
                fallback_return = (
                    self.generate_metal_vertex_output_position_fallback_return(
                        value, indent_str
                    )
                )
                if fallback_return is not None:
                    return fallback_return
                return f"{indent_str}return " f"{value};\n"
        elif hasattr(stmt, "__class__") and "ExpressionStatementNode" in str(
            type(stmt)
        ):
            tail_return = self.generate_tail_expression_statement(stmt, indent)
            if tail_return is not None:
                return tail_return
            expr_code = self.generate_expression_statement(stmt)
            return self.generate_statement_code(expr_code, indent)
        else:
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def generate_tail_expression_statement(self, stmt, indent=0):
        if not getattr(stmt, "is_tail_expression", False):
            return None
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            return None

        indent_str = "    " * indent
        return_wrapper = self.current_function_return_wrapper
        if return_wrapper is not None:
            value = self.generate_expression_with_expected(
                stmt.expression,
                return_wrapper["source_type"],
            )
            return f"{indent_str}return {return_wrapper['struct_name']}{{{value}}};\n"

        value = self.generate_expression_with_expected(stmt.expression, return_type)
        return f"{indent_str}return {value};\n"

    def generate_statement_code(self, code, indent=0):
        indent_str = "    " * indent
        lines = [line.rstrip() for line in str(code).splitlines() if line.strip()]
        if not lines:
            return ""

        result = ""
        for line in lines:
            stripped = line.strip()
            terminator = (
                ""
                if stripped.endswith((";", "{", "}", ":"))
                or stripped.startswith(("case ", "default:"))
                else ";"
            )
            result += f"{indent_str}{line}{terminator}\n"
        return result

    def local_variable_declared_type(self, stmt):
        var_type = getattr(stmt, "var_type", None)
        if var_type is None:
            var_type = getattr(stmt, "vtype", None)
        if var_type is None:
            var_type = self.expression_result_type(getattr(stmt, "initial_value", None))
        return self.type_name_string(var_type) or "float"

    def local_variable_texture_alias_resource_type(self, node, declared_type):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return None

        initial_type = self.texture_argument_resource_type(initial_value)
        if initial_type is None:
            return None

        declared_type_name = self.type_name_string(declared_type)
        if not self.is_texture_or_image_resource_type(declared_type_name):
            if (
                self.local_variable_type_node(node) is None
                and explicit_image_access(node, self.attribute_value_to_string) is None
                and explicit_image_format(node, self.attribute_value_to_string) is None
            ):
                return initial_type
            return None

        declared_resource_type = self.map_resource_type_with_format(
            declared_type_name, node
        )
        self.validate_local_resource_alias_compatibility(
            node, declared_type_name, declared_resource_type, initial_type
        )
        if (
            explicit_image_access(node, self.attribute_value_to_string) is None
            and explicit_image_format(node, self.attribute_value_to_string) is None
        ):
            return initial_type
        if explicit_image_access(node, self.attribute_value_to_string) is None:
            return initial_type
        return declared_resource_type

    def validate_local_resource_alias_compatibility(
        self, node, declared_type_name, declared_resource_type, initial_type
    ):
        alias_name = getattr(node, "name", "<anonymous>")
        declared_compat_type = self.local_resource_alias_compatibility_type(
            declared_resource_type
        )
        initial_compat_type = self.local_resource_alias_compatibility_type(initial_type)
        if declared_compat_type != initial_compat_type:
            raise ValueError(
                f"Metal local resource alias '{alias_name}' declares "
                f"{declared_type_name} but initializer has {initial_type}"
            )

        declared_access = explicit_image_access(node, self.attribute_value_to_string)
        if declared_access is not None:
            initial_access = self.storage_image_access_mode(initial_type)
            if initial_access is not None and declared_access != initial_access:
                raise ValueError(
                    f"Metal local resource alias '{alias_name}' declares "
                    f"access::{declared_access} but initializer has "
                    f"access::{initial_access}"
                )

        declared_format = explicit_image_format(node, self.attribute_value_to_string)
        if declared_format is not None:
            initial_format = self.image_resource_format(
                getattr(node, "initial_value", None)
            )
            if initial_format is not None and declared_format != initial_format:
                raise ValueError(
                    f"Metal local resource alias '{alias_name}' declares "
                    f"format {declared_format} but initializer has {initial_format}"
                )

    def local_resource_alias_compatibility_type(self, resource_type):
        resource_type = self.resource_base_type(resource_type)
        if self.is_storage_image_resource(resource_type):
            return self.storage_image_access_agnostic_type(resource_type)
        return resource_type

    def local_variable_texture_alias_declaration_type(self, node, resource_type):
        array_size = self.texture_argument_resource_array_size(
            getattr(node, "initial_value", None)
        )
        if array_size is None:
            return resource_type
        return self.format_resource_array_type(resource_type, array_size)

    def texture_argument_resource_array_size(self, texture_arg):
        if isinstance(texture_arg, ArrayAccessNode):
            return None
        member_array_size = self.struct_member_resource_array_size(texture_arg)
        if member_array_size is not None:
            return member_array_size
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None

        local_type = self.local_variable_types.get(texture_name)
        _element_type, array_size = self.metal_array_type_parts(local_type)
        if array_size is not None:
            return array_size

        if texture_name in self.current_texture_parameter_array_sizes:
            return self.current_texture_parameter_array_sizes[texture_name]

        for texture_variable, _, _texture_type, array_size in self.texture_variables:
            if getattr(texture_variable, "name", None) == texture_name:
                return array_size
        return None

    def format_resource_array_type(self, resource_type, array_size):
        return f"array<{resource_type}, {array_size or '1'}>"

    def metal_array_type_parts(self, type_name):
        if type_name is None:
            return None, None
        type_text = str(type_name).strip()
        if not (type_text.startswith("array<") and type_text.endswith(">")):
            return None, None

        inner = type_text[len("array<") : -1]
        depth = 0
        for index, char in enumerate(inner):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                return inner[:index].strip(), inner[index + 1 :].strip()
        return inner.strip(), None

    def metal_array_element_type(self, type_name):
        element_type, _array_size = self.metal_array_type_parts(type_name)
        return element_type

    def local_variable_sampler_alias_resource_type(self, node, declared_type):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return None

        initial_name = self.expression_name(initial_value)
        initial_type = self.expression_result_type(initial_value)
        initial_array_element_type = self.metal_array_element_type(initial_type)
        if not (
            self.is_sampler_type(initial_type)
            or self.is_sampler_type(initial_array_element_type)
            or initial_name in self.sampler_variable_names()
            or self.struct_member_is_sampler_resource(
                self.struct_resource_member_node(initial_value)
            )
        ):
            return None

        declared_type_name = self.type_name_string(declared_type)
        if self.local_variable_type_node(node) is None:
            array_size = self.sampler_argument_resource_array_size(initial_value)
            if array_size is not None:
                return self.format_resource_array_type("sampler", array_size)
            return "sampler"
        if self.is_sampler_type(declared_type_name):
            return "sampler"
        return None

    def local_variable_acceleration_structure_alias_resource_type(
        self, node, declared_type
    ):
        initial_type = self.acceleration_structure_argument_resource_type(
            getattr(node, "initial_value", None)
        )
        if initial_type is None:
            return None

        declared_type_name = self.type_name_string(declared_type)
        if self.local_variable_type_node(node) is None:
            return initial_type
        if self.is_acceleration_structure_type(declared_type_name):
            declared_resource_type = self.map_resource_type_with_format(
                declared_type_name, node
            )
            if declared_resource_type != initial_type:
                alias_name = getattr(node, "name", "<anonymous>")
                raise ValueError(
                    f"Metal local acceleration structure alias '{alias_name}' "
                    f"declares {declared_type_name} but initializer has "
                    f"{initial_type}"
                )
            return declared_resource_type
        return None

    def acceleration_structure_argument_resource_type(self, argument):
        if argument is None:
            return None

        argument_type = self.expression_result_type(argument)
        if argument_type and self.is_acceleration_structure_type(argument_type):
            if self.metal_type_is_pointer_like(argument_type):
                return None
            return self.map_resource_type_with_format(argument_type)

        argument_name = self.expression_name(argument)
        if not argument_name:
            return None

        local_type = self.local_variable_types.get(argument_name)
        if local_type and self.is_acceleration_structure_type(local_type):
            if self.metal_type_is_pointer_like(local_type):
                return None
            return self.map_resource_type_with_format(local_type)

        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            array_size,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) != argument_name:
                continue
            if array_size is not None and not isinstance(argument, ArrayAccessNode):
                return None
            return mapped_type
        return None

    def sampler_argument_resource_array_size(self, sampler_arg):
        if isinstance(sampler_arg, ArrayAccessNode):
            return None

        member_array_size = self.struct_member_resource_array_size(sampler_arg)
        if member_array_size is not None:
            return member_array_size

        sampler_name = self.expression_name(sampler_arg)
        if not sampler_name:
            return None

        local_type = self.local_variable_types.get(sampler_name)
        _element_type, array_size = self.metal_array_type_parts(local_type)
        if array_size is not None:
            return array_size

        if sampler_name in self.current_sampler_parameter_array_sizes:
            return self.current_sampler_parameter_array_sizes[sampler_name]

        for sampler_variable, _, array_size in self.sampler_variables:
            if getattr(sampler_variable, "name", None) == sampler_name:
                return array_size
        return None

    def sampler_argument_resource_type(self, sampler_arg):
        sampler_name = self.expression_name(sampler_arg)
        if sampler_name in self.sampler_variable_names():
            return "sampler"

        arg_type = self.expression_result_type(sampler_arg)
        array_element_type = self.metal_array_element_type(arg_type)
        if self.is_sampler_type(arg_type) or self.is_sampler_type(array_element_type):
            return "sampler"

        if self.struct_member_is_sampler_resource(
            self.struct_resource_member_node(sampler_arg)
        ):
            return "sampler"
        return None

    def record_local_texture_alias_sampler_source(self, node):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            self.current_texture_alias_sources[node.name] = initial_value

    def record_local_image_alias_metadata(self, node):
        image_format = explicit_image_format(node, self.attribute_value_to_string)
        if image_format is None:
            image_format = self.image_resource_format(
                getattr(node, "initial_value", None)
            )
        if image_format is not None:
            self.current_image_format_parameters[node.name] = image_format

    def local_variable_type_node(self, stmt):
        return getattr(stmt, "var_type", None) or getattr(stmt, "vtype", None)

    def local_variable_is_address_space_alias(self, node):
        return isinstance(
            self.local_variable_type_node(node), (PointerType, ReferenceType)
        )

    def local_variable_is_array_alias(self, node):
        raw_type = self.local_variable_type_node(node)
        if self.is_array_type_node(raw_type):
            return True
        type_name = self.type_name_string(raw_type)
        if type_name is None:
            type_name = self.local_variable_declared_type(node)
        return "[" in str(type_name) and str(type_name).rstrip().endswith("]")

    def local_pointer_initializer_needs_address(self, node, initial_value):
        if not isinstance(self.local_variable_type_node(node), PointerType):
            return False
        if initial_value is None:
            return False
        if (
            isinstance(initial_value, UnaryOpNode)
            and getattr(initial_value, "operator", getattr(initial_value, "op", None))
            == "&"
        ):
            return False
        if self.metal_type_is_pointer_like(self.expression_result_type(initial_value)):
            return False
        return self.assignment_target_root_name(initial_value) is not None

    def pointer_assignment_needs_address(self, target, value):
        target_type = self.expression_result_type(target)
        if self.pointer_pointee_type_name(target_type) is None:
            return False
        if value is None:
            return False
        if (
            isinstance(value, UnaryOpNode)
            and getattr(value, "operator", getattr(value, "op", None)) == "&"
        ):
            return False
        if self.metal_type_is_pointer_like(self.expression_result_type(value)):
            return False
        return self.assignment_target_root_name(value) is not None

    def address_space_assignment_diagnostic(self, target, value):
        target_type = self.expression_result_type(target)
        if self.pointer_pointee_type_name(target_type) is None:
            return None

        expected_address_space = self.argument_address_space(target)
        if expected_address_space is None:
            return None

        target_name = self.assignment_target_display_name(target) or "<target>"
        conflict = self.argument_address_space_conflict(value)
        if conflict is not None:
            return (
                "/* unsupported Metal address-space assignment: value "
                f"{self.address_space_conflict_description(conflict)} use "
                f"different address spaces; assignment to '{target_name}' "
                f"requires {expected_address_space} */"
            )

        actual_address_space = self.argument_address_space(value)
        if (
            actual_address_space is None
            or actual_address_space == expected_address_space
        ):
            return None

        value_name = self.assignment_target_display_name(value) or "<expr>"
        return (
            "/* unsupported Metal address-space assignment: value "
            f"'{value_name}' uses {actual_address_space} address space but "
            f"target '{target_name}' uses {expected_address_space} */"
        )

    def local_reference_assignment_target(self, target):
        if not isinstance(target, BinaryOpNode):
            return None
        if getattr(target, "operator", getattr(target, "op", None)) != "&":
            return None
        type_name = self.expression_name(getattr(target, "left", None))
        name = self.expression_name(getattr(target, "right", None))
        if not type_name or not name:
            return None
        if type_name in self.local_variable_types:
            return None
        if (
            type_name not in self.type_mapping
            and type_name not in self.struct_member_types
        ):
            return None
        return type_name, name

    def generate_local_reference_assignment_declaration(self, target, value, rhs):
        reference_target = self.local_reference_assignment_target(target)
        if reference_target is None:
            return None

        type_name, name = reference_target
        address_space = self.argument_address_space(value) or "thread"
        self.local_variable_types[name] = f"{type_name}&"
        self.current_address_space_variables[name] = address_space

        readonly_key = self.readonly_metal_mesh_payload_key(value)
        root_name = self.assignment_target_root_name(value)
        qualifier = address_space
        if readonly_key is not None:
            reason = self.current_readonly_metal_mesh_payload_reasons[readonly_key]
            self.current_readonly_metal_mesh_payload_parameters.add(name)
            self.current_readonly_metal_mesh_payload_reasons[name] = reason
            qualifier = "const object_data"
        elif root_name in self.current_readonly_metal_parameters:
            reason = self.current_readonly_metal_parameter_reasons[root_name]
            self.current_readonly_metal_parameters.add(name)
            self.current_readonly_metal_parameter_reasons[name] = reason
            if address_space != "constant":
                qualifier = f"const {address_space}"

        mapped_type = self.map_type(type_name)
        local_name = self.metal_local_identifier_name(name)
        return f"{qualifier} {mapped_type}& {local_name} = {rhs}"

    def collect_metal_lowered_program_scope_groupshared_globals(
        self, ast, target_stage, global_vars, all_functions
    ):
        groupshared_globals = [
            node
            for node in global_vars or []
            if self.is_program_scope_groupshared_global(node)
        ]
        if not groupshared_globals:
            return {}

        entries = self.metal_program_scope_groupshared_stage_entries(ast, target_stage)
        entry_ids = {id(func) for _stage_name, func, _stage_locals in entries}
        lowered_by_function = {}

        for node in groupshared_globals:
            name = self.metal_variable_name(node)
            if not name or self.program_scope_groupshared_initializer(node) is not None:
                continue

            referenced_by_helper = any(
                id(func) not in entry_ids
                and self.node_uses_identifier(getattr(func, "body", []), name)
                for func in all_functions or []
            )
            if referenced_by_helper:
                continue

            used_entries = []
            blocked_entry_reference = False
            for stage_name, func, stage_locals in entries:
                if not self.node_uses_identifier(getattr(func, "body", []), name):
                    continue
                if stage_name != "compute" or name in self.metal_entry_local_names(
                    func, stage_locals
                ):
                    blocked_entry_reference = True
                    break
                used_entries.append(func)

            if blocked_entry_reference:
                continue
            for func in used_entries:
                lowered_by_function.setdefault(id(func), []).append(node)

        return lowered_by_function

    def metal_program_scope_groupshared_stage_entries(self, ast, target_stage):
        entries = []
        seen = set()

        def add(stage_name, func, stage_local_variables=None):
            if func is None or id(func) in seen:
                return
            normalized_stage_name = normalize_stage_name(stage_name)
            if not normalized_stage_name:
                return
            if not stage_matches(target_stage, normalized_stage_name):
                return
            seen.add(id(func))
            entries.append(
                (normalized_stage_name, func, list(stage_local_variables or []))
            )

        for func in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(func)
            if stage_name in self.stage_entry_types():
                add(stage_name, func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            add(
                normalize_stage_name(stage_type),
                getattr(stage, "entry_point", None),
                getattr(stage, "local_variables", []),
            )

        return entries

    def metal_stage_local_variables_with_lowered_program_groupshared(
        self, func, stage_local_variables=None
    ):
        lowered_globals = (
            self.metal_lowered_program_scope_groupshared_globals_by_function.get(
                id(func), []
            )
        )
        if not lowered_globals:
            return stage_local_variables
        return list(lowered_globals) + list(stage_local_variables or [])

    def metal_entry_local_names(self, func, stage_local_variables=None):
        names = {
            getattr(variable, "name", None)
            for variable in stage_local_variables or []
            if getattr(variable, "name", None)
        }
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, (VariableNode, BackendVariableNode)):
                name = getattr(node, "name", None)
                if name:
                    names.add(name)
        return names

    def metal_variable_name(self, node):
        return getattr(node, "name", getattr(node, "variable_name", None))

    def program_scope_groupshared_initializer(self, node):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            return initial_value
        return getattr(node, "value", None)

    def global_variable_qualifier(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for address_space in (
            "constant",
            "device",
            "threadgroup_imageblock",
            "threadgroup",
            "thread",
        ):
            if address_space in qualifiers:
                return f"{address_space} "
        if qualifiers & {"const", "readonly"}:
            return "constant "
        return "constant "

    def is_program_scope_groupshared_global(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        }
        return bool(
            (qualifiers | attributes)
            & {"groupshared", "shared", "threadgroup", "workgroup"}
        )

    def record_metal_program_scope_value_global(
        self, name, vtype, qualifier, node=None
    ):
        if not name or not self.global_value_variable_requires_initializer(qualifier):
            return
        self.metal_program_scope_value_globals.add(name)
        if self.is_program_scope_groupshared_global(node):
            self.metal_program_scope_groupshared_globals.add(name)
        self.metal_program_scope_value_global_types[name] = self.type_name_string(vtype)

    def global_value_variable_requires_initializer(self, qualifier):
        return str(qualifier or "").strip().startswith("constant")

    def metal_program_scope_global_default_initializer(
        self, mapped_type, array_suffix=""
    ):
        _base_type, mapped_array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix or mapped_array_suffix:
            return "{}"
        return self.metal_default_value_expression(mapped_type)

    def metal_program_scope_global_initializer_diagnostic(self, name, node=None):
        if self.is_program_scope_groupshared_global(node):
            return (
                "/* unsupported Metal program-scope groupshared global: "
                f"'{name}' cannot be emitted as Metal threadgroup storage at "
                "program scope; using constant zero placeholder */"
            )
        return (
            "/* unsupported Metal program-scope global initializer: "
            f"'{name}' needs an initializer in the constant address space; "
            "using zero initializer */"
        )

    def local_variable_qualifier(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        }
        if self.local_variable_is_address_space_alias(node):
            raw_type = self.local_variable_type_node(node)
            memory_qualifiers = self.resource_memory_qualifier_prefix(node, raw_type)
            address_space = self.local_variable_address_space(node)
            if address_space is not None:
                const_prefix = ""
                if address_space != "constant" and (
                    "const" in qualifiers
                    or "readonly" in qualifiers
                    or self.local_variable_inherits_readonly_mesh_payload(node)
                    or self.local_variable_address_space_mismatch_diagnostic(node)
                    is not None
                ):
                    const_prefix = "const "
                return f"{memory_qualifiers}{const_prefix}{address_space} "
        if "threadgroup_imageblock" in qualifiers | attributes:
            return "threadgroup_imageblock "
        if (qualifiers | attributes) & {
            "shared",
            "groupshared",
            "threadgroup",
            "workgroup",
        }:
            return "threadgroup "
        if self.is_metal_atomic_value_type(self.local_variable_declared_type(node)):
            return "threadgroup "
        if self.local_variable_uses_constexpr_qualifier(node, qualifiers):
            return "constexpr "
        if qualifiers & {"const", "constant"}:
            return "const "
        return ""

    def local_variable_uses_constexpr_qualifier(self, node, qualifiers):
        if "constant" not in qualifiers:
            return False
        if self.local_variable_is_address_space_alias(node):
            return False
        if self.local_variable_is_array_alias(node):
            return False

        type_name = self.local_variable_declared_type(node)
        mapped_type = self.map_type(type_name)
        if mapped_type not in {"int", "uint"}:
            return False
        return (
            self.literal_int_value(
                getattr(node, "initial_value", None), self.literal_int_constants
            )
            is not None
        )

    def local_variable_address_space(self, node):
        explicit_address_space = self.normalized_address_space(
            self.parameter_address_space(node)
        )
        initializer_address_space = self.local_variable_initializer_address_space(node)
        if (
            self.local_variable_is_address_space_alias(node)
            and explicit_address_space is not None
            and initializer_address_space is not None
            and explicit_address_space != initializer_address_space
        ):
            return initializer_address_space
        if explicit_address_space is not None:
            return explicit_address_space
        if (
            self.local_variable_is_address_space_alias(node)
            and initializer_address_space is not None
        ):
            return initializer_address_space
        if self.is_metal_atomic_value_type(self.local_variable_declared_type(node)):
            return "threadgroup"
        return "thread"

    def local_variable_initializer_address_space(self, node):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return None
        return self.argument_address_space(initial_value)

    def local_variable_inherits_readonly_mesh_payload(self, node):
        if not self.local_variable_is_address_space_alias(node):
            return False
        initial_value = getattr(node, "initial_value", None)
        return self.readonly_metal_mesh_payload_key(initial_value) is not None

    def local_variable_address_space_mismatch_diagnostic(self, node):
        if not self.local_variable_is_address_space_alias(node):
            return None
        explicit_address_space = self.normalized_address_space(
            self.parameter_address_space(node)
        )
        initializer_address_space = self.local_variable_initializer_address_space(node)
        if (
            explicit_address_space is None
            or initializer_address_space is None
            or explicit_address_space == initializer_address_space
        ):
            return None
        initial_value = getattr(node, "initial_value", None)
        initializer_name = self.assignment_target_root_name(initial_value) or "<expr>"
        return (
            "/* unsupported Metal address-space local alias: initializer "
            f"'{initializer_name}' uses {initializer_address_space} address space "
            f"but local '{node.name}' was declared {explicit_address_space}; "
            f"using read-only {initializer_address_space} alias */"
        )

    def local_variable_address_space_conflict_diagnostic(self, node):
        raw_type = self.local_variable_type_node(node)
        if not isinstance(raw_type, (PointerType, ReferenceType)):
            return None
        conflict = self.argument_address_space_conflict(
            getattr(node, "initial_value", None)
        )
        if conflict is None:
            return None
        target_address_space = self.local_variable_address_space(node)
        fallback_kind = "value" if isinstance(raw_type, ReferenceType) else "alias"
        return (
            "/* unsupported Metal address-space local alias: initializer "
            f"{self.address_space_conflict_description(conflict)} use different "
            f"address spaces; using uninitialized {target_address_space} "
            f"{fallback_kind} */"
        )

    def local_variable_address_space_conflict_declaration(self, node, declaration):
        raw_type = self.local_variable_type_node(node)
        if not isinstance(raw_type, ReferenceType):
            return declaration
        address_space = self.local_variable_address_space(node)
        referent_type = self.map_resource_type_with_format(
            raw_type.referenced_type, node
        )
        value_declaration = format_c_style_array_declaration(
            referent_type, self.metal_local_identifier_name(node.name)
        )
        return self.maybe_format_unused_local_declaration(
            f"{address_space} {value_declaration}",
            node.name,
        )

    def record_readonly_metal_mesh_payload_local_alias(self, node):
        if self.local_variable_address_space(node) != "object_data":
            return
        if not self.local_variable_is_address_space_alias(node):
            return

        qualifiers = self.parameter_qualifier_names(node)
        reason = None
        if self.local_variable_address_space_mismatch_diagnostic(node) is not None:
            reason = "address-space mismatch alias"
        elif qualifiers & {"const", "constant"}:
            reason = "const-qualified payload parameter"
        else:
            initial_value = getattr(node, "initial_value", None)
            readonly_key = self.readonly_metal_mesh_payload_key(initial_value)
            reason = self.current_readonly_metal_mesh_payload_reasons.get(readonly_key)

        if reason is None:
            return
        self.current_readonly_metal_mesh_payload_parameters.add(node.name)
        self.current_readonly_metal_mesh_payload_reasons[node.name] = reason

    def readonly_metal_local_alias_reason(self, node):
        is_address_space_alias = self.local_variable_is_address_space_alias(node)
        is_array_alias = self.local_variable_is_array_alias(node)
        qualifiers = self.parameter_qualifier_names(node)
        if not (is_address_space_alias or is_array_alias):
            if "const" in qualifiers:
                return "const-qualified local"
            if "constant" in qualifiers:
                return "constant-qualified local"
            return None
        if (
            is_address_space_alias
            and self.local_variable_address_space(node) == "object_data"
        ):
            return None
        if "const" in qualifiers and is_array_alias:
            return "const-qualified local array"
        if "const" in qualifiers:
            return "const-qualified local alias"
        if "constant" in qualifiers and is_array_alias:
            return "constant-qualified local array"
        if "constant" in qualifiers:
            return "constant address-space local alias"
        if "readonly" in qualifiers:
            return "readonly local alias"
        return None

    def record_readonly_metal_local_alias(self, node):
        reason = self.readonly_metal_local_alias_reason(node)
        if reason is None:
            return
        self.current_readonly_metal_parameters.add(node.name)
        self.current_readonly_metal_parameter_reasons[node.name] = reason

    def is_metal_atomic_value_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if type_name is None:
            return False
        type_name = type_name.split("[", 1)[0].strip()
        return type_name in {
            "atomic_bool",
            "atomic_int",
            "atomic_uint",
        }

    def metal_atomic_array_type_parts(self, vtype):
        type_name = self.type_name_string(vtype)
        if type_name is None or "[" not in type_name:
            return None

        atomic_type = type_name.split("[", 1)[0].strip()
        if not self.is_metal_atomic_value_type(atomic_type):
            return None

        size = None
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            size = evaluate_literal_int_expression(
                getattr(vtype, "size", None),
                self.literal_int_constants,
            )
        if size is None:
            _base_type, size = parse_array_type(type_name)
        if size is None:
            size = self.literal_int_array_size_from_type_name(type_name)
        return atomic_type, size

    def literal_int_array_size_from_type_name(self, type_name):
        if not isinstance(type_name, str) or "[" not in type_name:
            return None

        _base_type, array_suffix = split_array_type_suffix(type_name)
        close_bracket = array_suffix.find("]")
        if close_bracket == -1:
            return None

        size_expr = array_suffix[1:close_bracket].strip()
        if not size_expr:
            return None
        return self.literal_int_resource_array_size(size_expr)

    def metal_atomic_value_type(self, atomic_type):
        return {
            "atomic_bool": "bool",
            "atomic_int": "int",
            "atomic_uint": "uint",
        }.get(str(atomic_type), atomic_type)

    def metal_atomic_zero_value(self, atomic_type):
        return {
            "atomic_bool": "false",
            "atomic_int": "0",
            "atomic_uint": "0u",
        }.get(str(atomic_type), "0")

    def generate_metal_atomic_array_initializer_stores(
        self, name, var_type, initial_value, indent_str
    ):
        array_parts = self.metal_atomic_array_type_parts(var_type)
        if array_parts is None or not isinstance(initial_value, ArrayLiteralNode):
            return None

        atomic_type, declared_size = array_parts
        elements = list(getattr(initial_value, "elements", []) or [])
        store_count = declared_size if declared_size is not None else len(elements)
        value_type = self.metal_atomic_value_type(atomic_type)
        zero_value = self.metal_atomic_zero_value(atomic_type)
        lines = []
        for index in range(store_count):
            if index < len(elements):
                value = self.generate_expression_with_expected(
                    elements[index], value_type
                )
            else:
                value = zero_value
            lines.append(
                f"{indent_str}atomic_store_explicit(&{name}[{index}], {value}, "
                "memory_order_relaxed);\n"
            )
        if declared_size is not None and len(elements) > declared_size:
            lines.append(
                f"{indent_str}/* unsupported Metal atomic array initializer: "
                f"'{name}' has {len(elements)} initializers for {declared_size} "
                "elements; extra values were ignored */\n"
            )
        return "".join(lines)

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if (
            hasattr(vtype, "name")
            or hasattr(vtype, "element_type")
            or isinstance(vtype, (PointerType, ReferenceType))
        ):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

    def option_payload_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        if not isinstance(type_name, str):
            return None

        base_name, generic_args = generic_type_parts(type_name.strip())
        if base_name.rsplit("::", 1)[-1] != "Option" or len(generic_args) != 1:
            return None
        return generic_args[0]

    def lowerable_option_payload_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        payload_type = self.option_payload_type_name(type_name)
        if payload_type is None:
            return None
        if generic_enum_specialized_type_name(self, type_name) is not None:
            return None
        if generic_struct_specialized_type_name(self, type_name) is not None:
            return None
        return payload_type

    def expected_option_payload_type_name(self):
        expected_type = self.current_expression_expected_type
        option_payload = self.lowerable_option_payload_type_name(
            self.current_expression_expected_type
        ) or self.lowerable_option_payload_type_name(self.current_function_return_type)
        if option_payload is not None:
            return option_payload
        if self.option_payload_type_name(expected_type) is not None:
            return None
        return expected_type

    def option_constructor_payload_expression(self, expr):
        func_expr = getattr(expr, "function", None)
        if func_expr is None:
            func_expr = getattr(expr, "name", None)
        func_name = getattr(func_expr, "name", func_expr)
        if not isinstance(func_name, str):
            return None
        if func_name.rsplit("::", 1)[-1] != "Some":
            return None

        args = list(getattr(expr, "args", getattr(expr, "arguments", [])) or [])
        payload_type = self.expected_option_payload_type_name()
        if payload_type is None:
            return None
        if args:
            return self.generate_expression_with_expected(args[0], payload_type)
        return self.metal_default_value_expression(payload_type)

    def option_none_default_expression(self):
        payload_type = self.expected_option_payload_type_name()
        if payload_type is None:
            return None
        return self.metal_default_value_expression(payload_type)

    def is_metal_ray_query_type_name(self, type_name):
        type_name = str(type_name or "").strip()
        return (
            type_name == "RayQuery"
            or type_name == "rayQueryEXT"
            or type_name.startswith("RayQuery<")
        )

    def requires_metal_builtin_ray_desc(self, type_name):
        type_name = str(type_name or "").strip()
        return type_name == "RayDesc" and type_name not in getattr(
            self, "structs_by_name", {}
        )

    def require_metal_ray_query_runtime(self):
        self.required_metal_ray_query_runtime = True

    def metal_ray_query_method_return_type(self, operation):
        return self.METAL_RAY_QUERY_RETURN_TYPES.get(str(operation))

    def is_metal_ray_query_method(self, operation):
        return self.metal_ray_query_method_return_type(operation) is not None

    def metal_ray_query_helper_name(self, operation):
        return f"cgl_ray_query_{self.metal_snake_case_name(operation)}"

    def metal_snake_case_name(self, name):
        text = str(name)
        result = []
        for index, char in enumerate(text):
            if char.isupper() and index > 0:
                previous = text[index - 1]
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if previous.islower() or next_char.islower():
                    result.append("_")
            result.append(char.lower())
        return "".join(result)

    def ray_query_operation_from_function_call(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return None
        func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
        if not isinstance(func_expr, MemberAccessNode):
            return None
        operation = str(getattr(func_expr, "member", ""))
        if not self.is_metal_ray_query_method(operation):
            return None
        receiver_type = self.expression_result_type(
            getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        )
        if receiver_type is not None and self.is_metal_ray_query_type_name(
            self.type_name_string(receiver_type)
        ):
            return operation
        return None

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def is_scalar_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float",
            "half",
            "double",
            "int",
            "uint",
            "bool",
        }

    def is_vector_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float2",
            "float3",
            "float4",
            "half2",
            "half3",
            "half4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
            "bool2",
            "bool3",
            "bool4",
        }

    def is_matrix_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        mapped_type, array_suffix = split_array_type_suffix(self.map_type(vtype))
        if array_suffix:
            return False
        for prefix in ("float", "half", "double"):
            if not mapped_type.startswith(prefix):
                continue
            dimensions = mapped_type[len(prefix) :].split("x", 1)
            return (
                len(dimensions) == 2
                and dimensions[0] in {"2", "3", "4"}
                and dimensions[1] in {"2", "3", "4"}
            )
        return False

    def is_builtin_value_constructor_type(self, vtype):
        return (
            self.is_scalar_value_type(vtype)
            or self.is_vector_value_type(vtype)
            or self.is_matrix_value_type(vtype)
        )

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("half"):
            return "half"
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type.startswith("bool"):
            return "bool"
        return None

    def vector_value_width(self, vtype):
        mapped_type = self.map_type(vtype)
        if not mapped_type or not mapped_type[-1:].isdigit():
            return None
        if self.is_matrix_value_type(mapped_type):
            return None
        if self.vector_component_type(mapped_type) is None:
            return None
        return int(mapped_type[-1])

    def cast_binary_operand_to_expected_vector(self, code, operand_type, expected_type):
        if operand_type is None:
            return code
        expected_mapped = self.map_type(expected_type)
        expected_width = self.vector_value_width(expected_mapped)
        if expected_width is None:
            return code

        operand_mapped = self.map_type(operand_type)
        if operand_mapped == expected_mapped:
            return code
        if self.is_scalar_value_type(operand_mapped):
            return f"{expected_mapped}({code})"
        if self.vector_value_width(operand_mapped) == expected_width:
            return f"{expected_mapped}({code})"
        return code

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, IdentifierNode):
            builtin_type = self.metal_graphics_builtin_result_type(expr.name)
            if builtin_type is not None:
                return builtin_type
            builtin_type = self.metal_compute_builtin_result_type(expr.name)
            if builtin_type is not None:
                return builtin_type
            return self.local_variable_types.get(
                expr.name
            ) or self.metal_program_scope_value_global_types.get(expr.name)
        if isinstance(expr, (VariableNode, BackendVariableNode)):
            name = getattr(expr, "name", None)
            builtin_type = self.metal_graphics_builtin_result_type(name)
            if builtin_type is not None:
                return builtin_type
            builtin_type = self.metal_compute_builtin_result_type(name)
            if builtin_type is not None:
                return builtin_type
            return self.local_variable_types.get(
                name
            ) or self.metal_program_scope_value_global_types.get(name)
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            operator = self.map_operator(getattr(expr, "op", ""))
            if operator in {"<", ">", "<=", ">=", "==", "!=", "&&", "||"}:
                for candidate_type in (left_type, right_type):
                    mapped_type = self.map_type(candidate_type)
                    component_type = self.vector_component_type(mapped_type)
                    if component_type is None:
                        continue
                    width = mapped_type[-1] if mapped_type[-1:].isdigit() else ""
                    if width:
                        return f"bool{width}"
                return "bool"
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            operand_type = self.expression_result_type(expr.operand)
            if getattr(expr, "operator", None) == "*":
                return self.pointer_pointee_type_name(operand_type) or operand_type
            return operand_type
        if isinstance(expr, TernaryOpNode):
            true_type = self.expression_result_type(getattr(expr, "true_expr", None))
            false_type = self.expression_result_type(getattr(expr, "false_expr", None))
            if self.is_vector_value_type(true_type):
                return true_type
            if self.is_vector_value_type(false_type):
                return false_type
            if true_type == false_type:
                return true_type
            if true_type == "float" or false_type == "float":
                return "float"
            return true_type or false_type
        if isinstance(expr, (AssignmentNode, BackendAssignmentNode)):
            return self.expression_result_type(
                getattr(expr, "target", getattr(expr, "left", None))
            )
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_result_type(expr.array)
            if not array_type:
                array_type = self.unsupported_glsl_buffer_block_expression_type(
                    expr.array
                )
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type
            metal_array_element_type = self.metal_array_element_type(array_type)
            if metal_array_element_type is not None:
                return metal_array_element_type
            pointee_type = self.pointer_pointee_type_name(array_type)
            if pointee_type is not None:
                return pointee_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            block_access = self.glsl_buffer_block_member_access(expr)
            if block_access is not None:
                return block_access["type"]
            object_type = self.expression_result_type(
                expr.object
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.object)
            object_type = self.member_lookup_type_name(object_type)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                component_type = self.vector_component_type(object_type)
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            member_types = {
                self.type_name_string(members[member])
                for members in self.struct_member_types.values()
                if member in members
            }
            if len(member_types) == 1:
                return next(iter(member_types))
            return None
        if isinstance(expr, PointerAccessNode):
            object_type = self.expression_result_type(
                expr.pointer_expr
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.pointer_expr)
            object_type = self.member_lookup_type_name(object_type)
            member = str(expr.member)
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            return None
        if isinstance(expr, ConstructorNode):
            return infer_enum_constructor_type(
                self, expr
            ) or infer_struct_constructor_type(self, expr)
        if isinstance(expr, MatchNode):
            return infer_match_expression_result_type(self, expr)
        if isinstance(expr, WaveOpNode):
            return self.metal_wave_result_type(expr.operation, expr.arguments)
        if isinstance(expr, RayQueryOpNode):
            return self.metal_ray_query_method_return_type(expr.operation)
        if isinstance(expr, FunctionCallNode):
            ray_query_operation = self.ray_query_operation_from_function_call(expr)
            if ray_query_operation is not None:
                return self.metal_ray_query_method_return_type(ray_query_operation)
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            if isinstance(func_name, str) and func_name.rsplit("::", 1)[-1] == "Some":
                payload_type = self.expected_option_payload_type_name()
                if payload_type is not None:
                    return payload_type
                if self.option_payload_type_name(self.current_expression_expected_type):
                    return None
                return self.expression_result_type(args[0]) if args else None
            if func_name == "RayDesc" and self.requires_metal_builtin_ray_desc(
                func_name
            ):
                return "RayDesc"
            if func_name in self.METAL_WAVE_INTRINSIC_ARITIES:
                return self.metal_wave_result_type(func_name, args)
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type:
                return numeric_result_type
            if func_name in {"normalize", "reflect"} and args:
                return self.expression_result_type(args[0])
            if func_name == "inverse" and args:
                return self.expression_result_type(args[0])
            if func_name == "transpose" and args:
                arg_type = self.expression_result_type(args[0])
                dimensions = self.metal_matrix_dimensions(arg_type)
                if dimensions is not None:
                    component_prefix, columns, rows = dimensions
                    return f"{component_prefix}{rows}x{columns}"
                return arg_type
            if func_name == "dot" and args:
                return (
                    self.vector_component_type(self.expression_result_type(args[0]))
                    or "float"
                )
            if func_name == "cross" and args:
                return self.expression_result_type(args[0])
            if (
                func_name == "mod"
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            if (
                func_name == "frac"
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            if (
                func_name == "lerp"
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            if (
                func_name in {"inverseSqrt", "inversesqrt"}
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            if (
                func_name in self.METAL_DERIVATIVE_FUNCTION_ALIASES
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            bitcast_result_type = self.metal_bitcast_result_type(func_name, args)
            if bitcast_result_type is not None:
                return bitcast_result_type
            integer_bit_result_type = self.metal_integer_bit_result_type(
                func_name, args
            )
            if integer_bit_result_type is not None:
                return integer_bit_result_type
            if func_name in {"mix", "clamp", "min", "max", "saturate"} and args:
                return self.expression_result_type(args[0])
            if is_resource_size_query_operation(func_name) and args:
                texture_type = self.texture_argument_resource_type(args[0])
                raw_type = self.expression_result_type(args[0])
                descriptor = self.texture_query_size_descriptor_for_argument(
                    texture_type, raw_type
                )
                if descriptor is not None:
                    return descriptor["return_type"]
            if is_texture_sampling_operation(func_name) and args:
                return self.texture_sample_result_type(args[0])
            specialized_func_name = generic_function_call_name(self, func_name, args)
            if specialized_func_name in getattr(self, "function_return_types", {}):
                return self.function_return_types[specialized_func_name]
            if func_name in getattr(self, "function_return_types", {}):
                return self.function_return_types[func_name]
            unsupported_functions = getattr(
                self, "unsupported_glsl_buffer_block_functions", {}
            )
            if func_name in unsupported_functions:
                return unsupported_functions[func_name].get("return_type")
            if func_name == "imageLoad" and args:
                return self.image_load_result_type(args[0])
            if func_name == "subpassLoad":
                return "vec4"
            if func_name in {
                "float",
                "half",
                "float16",
                "min16float",
                "min10float",
                "double",
                "f16",
                "f32",
                "f64",
                "int",
                "i8",
                "i16",
                "i32",
                "char",
                "signed char",
                "int8",
                "int16",
                "int8_t",
                "int16_t",
                "int32_t",
                "int64",
                "int64_t",
                "long",
                "signed long",
                "ptrdiff_t",
                "min16int",
                "min12int",
                "uint",
                "u8",
                "u16",
                "u32",
                "uchar",
                "unsigned char",
                "uint8",
                "uint16",
                "uint8_t",
                "uint16_t",
                "uint32_t",
                "uint64",
                "uint64_t",
                "ulong",
                "unsigned long",
                "size_t",
                "min16uint",
                "short",
                "signed short",
                "ushort",
                "unsigned short",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "vec2<f16>",
                "vec3<f16>",
                "vec4<f16>",
                "vec2<f32>",
                "vec3<f32>",
                "vec4<f32>",
                "vec2<f64>",
                "vec3<f64>",
                "vec4<f64>",
                "vec2<i8>",
                "vec3<i8>",
                "vec4<i8>",
                "vec2<u8>",
                "vec3<u8>",
                "vec4<u8>",
                "vec2<i16>",
                "vec3<i16>",
                "vec4<i16>",
                "vec2<u16>",
                "vec3<u16>",
                "vec4<u16>",
                "vec2<i32>",
                "vec3<i32>",
                "vec4<i32>",
                "vec2<u32>",
                "vec3<u32>",
                "vec4<u32>",
                "vec2<bool>",
                "vec3<bool>",
                "vec4<bool>",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "float2",
                "float3",
                "float4",
                "packed_float2",
                "packed_float3",
                "packed_float4",
                "simd_float2",
                "simd_float3",
                "simd_float4",
                "int2",
                "int3",
                "int4",
                "packed_int2",
                "packed_int3",
                "packed_int4",
                "simd_int2",
                "simd_int3",
                "simd_int4",
                "uint2",
                "uint3",
                "uint4",
                "packed_uint2",
                "packed_uint3",
                "packed_uint4",
                "simd_uint2",
                "simd_uint3",
                "simd_uint4",
                "bool2",
                "bool3",
                "bool4",
                "half2",
                "half3",
                "half4",
                "packed_half2",
                "packed_half3",
                "packed_half4",
                "simd_half2",
                "simd_half3",
                "simd_half4",
                "f16vec2",
                "f16vec3",
                "f16vec4",
                "mat2",
                "mat3",
                "mat4",
                "mat2x2",
                "mat2x3",
                "mat2x4",
                "mat3x2",
                "mat3x3",
                "mat3x4",
                "mat4x2",
                "mat4x3",
                "mat4x4",
                "f16mat2",
                "f16mat3",
                "f16mat4",
                "f16mat2x2",
                "f16mat2x3",
                "f16mat2x4",
                "f16mat3x2",
                "f16mat3x3",
                "f16mat3x4",
                "f16mat4x2",
                "f16mat4x3",
                "f16mat4x4",
                "simd_float2x2",
                "simd_float2x3",
                "simd_float2x4",
                "simd_float3x2",
                "simd_float3x3",
                "simd_float3x4",
                "simd_float4x2",
                "simd_float4x3",
                "simd_float4x4",
                "simd_half2x2",
                "simd_half2x3",
                "simd_half2x4",
                "simd_half3x2",
                "simd_half3x3",
                "simd_half3x4",
                "simd_half4x2",
                "simd_half4x3",
                "simd_half4x4",
                "char2",
                "char3",
                "char4",
                "uchar2",
                "uchar3",
                "uchar4",
                "i8vec2",
                "i8vec3",
                "i8vec4",
                "u8vec2",
                "u8vec3",
                "u8vec4",
                "short2",
                "short3",
                "short4",
                "ushort2",
                "ushort3",
                "ushort4",
                "i16vec2",
                "i16vec3",
                "i16vec4",
                "u16vec2",
                "u16vec3",
                "u16vec4",
                "min16float2",
                "min16float3",
                "min16float4",
                "min10float2",
                "min10float3",
                "min10float4",
                "min16int2",
                "min16int3",
                "min16int4",
                "min12int2",
                "min12int3",
                "min12int4",
                "min16uint2",
                "min16uint3",
                "min16uint4",
                "min16float2x2",
                "min16float2x3",
                "min16float2x4",
                "min16float3x2",
                "min16float3x3",
                "min16float3x4",
                "min16float4x2",
                "min16float4x3",
                "min16float4x4",
                "min10float2x2",
                "min10float2x3",
                "min10float2x4",
                "min10float3x2",
                "min10float3x3",
                "min10float3x4",
                "min10float4x2",
                "min10float4x3",
                "min10float4x4",
            }:
                return str(func_name)
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            value = getattr(expr, "value", None)
            if isinstance(value, float):
                return "float"
            if isinstance(value, int):
                return "int"
            if isinstance(value, str):
                return "float" if "." in value else "int"
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.local_variable_types.get(getattr(expr, "name", None))
        return None

    def generate_expression_statement(self, stmt):
        """Generate code for expression statements."""
        expr_node = getattr(stmt, "expression", stmt)
        discard_statement = self.generate_fragment_discard_statement(expr_node)
        if discard_statement is not None:
            return discard_statement

        if hasattr(stmt, "expression"):
            if isinstance(stmt.expression, (AssignmentNode, BackendAssignmentNode)):
                return self.generate_assignment(stmt.expression)
            expr = self.generate_expression(stmt.expression)
            return expr
        else:
            return self.generate_expression(stmt)

    def generate_fragment_discard_statement(self, expr):
        """Lower CrossGL/HLSL fragment-kill expressions to Metal syntax."""
        if isinstance(expr, IdentifierNode) and expr.name == "discard":
            return "discard_fragment()"

        if not isinstance(expr, FunctionCallNode):
            return None

        func_expr = getattr(expr, "function", getattr(expr, "name", None))
        func_name = getattr(func_expr, "name", func_expr)
        args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
        if func_name == "discard" and not args:
            return "discard_fragment()"
        if func_name != "clip" or len(args) != 1:
            return None

        predicate = self.generate_clip_discard_predicate(args[0])
        return f"if ({predicate}) {{\n    discard_fragment();\n}}"

    def generate_clip_discard_predicate(self, expr):
        """Return the Metal predicate for HLSL-style ``clip(expr)``."""
        rendered = self.generate_expression(expr)
        expr_type = self.expression_result_type(expr)
        width = self.vector_value_width(expr_type)
        if width is None:
            return f"({rendered}) < 0.0"

        mapped_type = self.map_type(expr_type)
        component_type = self.vector_component_type(mapped_type) or "float"
        zero = "0.0" if component_type in {"float", "half", "double"} else "0"
        return f"any(({rendered}) < {mapped_type}({zero}))"

    def generate_assignment(self, node):
        if hasattr(node, "target") and hasattr(node, "value"):
            target = node.target
            value = node.value
            rhs = self.generate_expression_with_expected(
                value, self.expression_result_type(target)
            )
            op = getattr(node, "operator", "=")
        else:
            target = node.left
            value = node.right
            rhs = self.generate_expression_with_expected(
                value, self.expression_result_type(target)
            )
            op = getattr(node, "operator", "=")

        reference_declaration = self.generate_local_reference_assignment_declaration(
            target, value, rhs
        )
        if reference_declaration is not None:
            return reference_declaration

        stage_io_assignment = self.generate_metal_stage_io_lowered_assignment(
            target, rhs, op
        )
        if stage_io_assignment is not None:
            return stage_io_assignment

        mesh_output_store = self.generate_metal_mesh_output_assignment(
            target, rhs, op, getattr(node, "value", getattr(node, "right", None))
        )
        if mesh_output_store is not None:
            return mesh_output_store

        block_store = self.generate_glsl_buffer_block_store(target, rhs, op)
        if block_store is not None:
            return block_store
        unsupported_store = self.unsupported_glsl_buffer_block_assignment_diagnostic(
            target
        )
        if unsupported_store is not None:
            return unsupported_store
        readonly_mesh_payload_store = (
            self.readonly_metal_mesh_payload_assignment_diagnostic(target)
        )
        if readonly_mesh_payload_store is not None:
            return readonly_mesh_payload_store
        readonly_raw_buffer_store = self.readonly_raw_buffer_assignment_diagnostic(
            target
        )
        if readonly_raw_buffer_store is not None:
            return readonly_raw_buffer_store
        readonly_parameter_store = self.readonly_metal_parameter_assignment_diagnostic(
            target
        )
        if readonly_parameter_store is not None:
            return readonly_parameter_store
        readonly_global_store = (
            self.readonly_metal_program_scope_global_assignment_diagnostic(target)
        )
        if readonly_global_store is not None:
            return readonly_global_store
        readonly_mesh_payload_alias = (
            self.readonly_metal_mesh_payload_alias_assignment_diagnostic(target, value)
        )
        if readonly_mesh_payload_alias is not None:
            return readonly_mesh_payload_alias
        address_space_assignment = self.address_space_assignment_diagnostic(
            target, value
        )
        if address_space_assignment is not None:
            return address_space_assignment

        lhs = self.generate_expression(target)
        if self.pointer_assignment_needs_address(target, value):
            rhs = f"&{rhs}"
        return f"{lhs} {op} {rhs}"

    def generate_metal_stage_io_lowered_assignment(self, target, rhs, op):
        if isinstance(target, MemberAccessNode):
            info = self.metal_stage_io_lowered_member_info(target)
            if info is None or info["lowering"].get("kind") != "matrix":
                return None
            lowering = info["lowering"]
            temp_name = self.next_metal_temp_variable("stage_io_matrix")
            lines = [f"{lowering['mapped_type']} {temp_name} = {rhs}"]
            for index, (field_name, _field_type) in enumerate(lowering["fields"]):
                lines.append(
                    f"{info['object_code']}.{field_name} {op} {temp_name}[{index}]"
                )
            return "\n".join(lines)

        if isinstance(target, ArrayAccessNode):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            if not isinstance(array_expr, MemberAccessNode):
                return None
            info = self.metal_stage_io_lowered_member_info(array_expr)
            if info is None or info["lowering"].get("kind") != "array":
                return None
            lowering = info["lowering"]
            index_expr = getattr(target, "index", None)
            literal_index = self.literal_int_value(
                index_expr, self.literal_int_constants
            )
            if literal_index is not None and 0 <= literal_index < lowering["size"]:
                field_name = lowering["fields"][literal_index][0]
                return f"{info['object_code']}.{field_name} {op} {rhs}"

            rendered_index = self.generate_expression(index_expr)
            temp_name = self.next_metal_temp_variable("stage_io_array")
            lines = [f"{lowering['element_type']} {temp_name} = {rhs}"]
            lines.append(f"switch ({rendered_index}) {{")
            for index, (field_name, _field_type) in enumerate(lowering["fields"]):
                lines.append(f"case {index}:")
                lines.append(
                    f"    {info['object_code']}.{field_name} {op} {temp_name};"
                )
                lines.append("    break;")
            lines.append("default:")
            lines.append("    break;")
            lines.append("}")
            return "\n".join(lines)

        return None

    def metal_stage_io_lowered_member_info(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        object_type = self.expression_result_type(object_expr)
        object_type = self.member_lookup_type_name(object_type)
        struct_name = self.type_name_string(object_type)
        member_name = str(getattr(expr, "member", ""))
        lowering = self.metal_stage_io_member_lowerings.get(struct_name, {}).get(
            member_name
        )
        if lowering is None:
            return None
        return {
            "struct_name": struct_name,
            "member_name": member_name,
            "object_code": self.generate_expression_with_expected(object_expr, None),
            "lowering": lowering,
        }

    def generate_metal_stage_io_lowered_member_access(self, expr):
        info = self.metal_stage_io_lowered_member_info(expr)
        if info is None or info["lowering"].get("kind") != "matrix":
            return None
        lowering = info["lowering"]
        columns = ", ".join(
            f"{info['object_code']}.{field_name}"
            for field_name, _field_type in lowering["fields"]
        )
        return f"{lowering['mapped_type']}({columns})"

    def generate_metal_stage_io_lowered_array_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        if not isinstance(array_expr, MemberAccessNode):
            return None
        info = self.metal_stage_io_lowered_member_info(array_expr)
        if info is None or info["lowering"].get("kind") != "array":
            return None
        lowering = info["lowering"]
        index_expr = getattr(expr, "index", None)
        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if literal_index is not None and 0 <= literal_index < lowering["size"]:
            field_name = lowering["fields"][literal_index][0]
            return f"{info['object_code']}.{field_name}"
        helper_name = self.metal_stage_io_array_getter_name(
            info["struct_name"], info["member_name"]
        )
        return f"{helper_name}({info['object_code']}, {self.generate_expression(index_expr)})"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        code += self.generate_scoped_statement_body(if_body, indent + 1)

        code += f"{indent_str}}}"

        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                elif_condition = self.generate_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f" else if ({elif_condition}) {{\n"

                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                code += self.generate_scoped_statement_body(elif_body, indent + 1)

                code += f"{indent_str}}}"

                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "else if"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if ("):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if (", " else if (", 1
                            )
                        code += "\n".join(
                            remaining_lines[1:]
                        )  # Skip first line as we already handled it
                    else:
                        code += " else {\n"
                        code += self.generate_scoped_statement_body(
                            nested_else, indent + 1
                        )
                        code += f"{indent_str}}}"
            else:
                code += " else {\n"
                code += self.generate_scoped_statement_body(else_branch, indent + 1)
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )

        try:
            init = self.generate_for_initializer(getattr(node, "init", None))

            condition = (
                self.generate_expression(node.condition)
                if getattr(node, "condition", None)
                else ""
            )

            update = (
                self.generate_expression(node.update)
                if getattr(node, "update", None)
                else ""
            )

            code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

            code += self.generate_scoped_statement_body(node.body, indent + 1)

            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def generate_block(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}{{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "statements", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_scoped_statement_body(self, body, indent):
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        try:
            return self.generate_statement_body(body, indent)
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )

        try:
            self.local_variable_types[pattern] = "int"
            self.current_unsupported_glsl_buffer_block_local_variables.discard(pattern)

            if isinstance(iterable_node, RangeNode):
                start = self.generate_expression(iterable_node.start)
                end = self.generate_expression(iterable_node.end)
                comparator = "<=" if iterable_node.inclusive else "<"
                code = (
                    f"{indent_str}for (int {pattern} = {start}; "
                    f"{pattern} {comparator} {end}; ++{pattern}) {{\n"
                )
            else:
                iterable = self.generate_expression(iterable_node)
                code = (
                    f"{indent_str}for (int {pattern} = 0; {pattern} < {iterable}; "
                    f"++{pattern}) {{\n"
                )

            code += self.generate_scoped_statement_body(
                getattr(node, "body", []), indent + 1
            )
            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(getattr(node, "condition", ""))

        code = f"{indent_str}while ({condition}) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(getattr(node, "condition", ""))

        code = f"{indent_str}do {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}} while ({condition});\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent

        code = f"{indent_str}while (true) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}switch ({expression}) {{\n"
        for case in getattr(node, "cases", []) or []:
            value = getattr(case, "value", None)
            if value is None:
                label = "default"
            else:
                label = f"case {self.generate_expression(value)}"
            code += self.generate_switch_case(
                label, getattr(case, "statements", []), indent + 1
            )

        default_case = getattr(node, "default_case", None)
        if default_case is not None:
            code += self.generate_switch_case(
                "default",
                getattr(default_case, "statements", default_case),
                indent + 1,
            )

        code += f"{indent_str}}}\n"
        return code

    def generate_match(self, node, indent):
        expression_type = self.type_name_string(
            self.expression_result_type(getattr(node, "expression", None))
        )
        if is_switch_lowerable_match(node) and expression_type != "bool":
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "Metal")

    def format_match_condition(self, condition):
        condition = str(condition).strip()
        if self.condition_has_single_outer_parentheses(condition):
            return condition
        return f"({condition})"

    def match_expression_requires_fallback_assignment(self):
        return True

    def generate_match_expression_diagnostic_assignment(
        self,
        target_variable,
        target_type,
        reason,
        indent,
        target_name,
    ):
        indent_str = "    " * indent
        fallback = self.metal_default_value_expression(target_type)
        return (
            f"{indent_str}{target_variable} = {fallback} "
            f"/* fallback for generated {target_name}: {reason} */;\n"
        )

    def condition_has_single_outer_parentheses(self, condition):
        if not (condition.startswith("(") and condition.endswith(")")):
            return False
        depth = 0
        for index, char in enumerate(condition):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(condition) - 1:
                    return False
            if depth < 0:
                return False
        return depth == 0

    def statement_body_terminates(self, body):
        if hasattr(body, "statements"):
            statements = body.statements
        elif isinstance(body, list):
            statements = body
        elif body is None:
            statements = []
        else:
            statements = [body]

        return bool(statements) and isinstance(
            statements[-1], (BreakNode, ContinueNode, ReturnNode)
        )

    def statement_body_has_statements(self, body):
        if hasattr(body, "statements"):
            return bool(body.statements)
        if isinstance(body, list):
            return bool(body)
        return body is not None

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = "    " * indent
        if not auto_break and not self.statement_body_has_statements(body):
            return f"{indent_str}{label}:\n"

        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}    break;\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_statement_body(self, body, indent):
        if hasattr(body, "statements"):
            statements = list(body.statements)
        elif isinstance(body, list):
            statements = list(body)
        elif body is not None:
            statements = [body]
        else:
            statements = []

        code = ""
        previous_unused_declarations = self.current_unused_local_declaration_names
        try:
            for index, stmt in enumerate(statements):
                self.current_unused_local_declaration_names = (
                    self.unused_local_declaration_names_for_statement(
                        stmt,
                        statements[index + 1 :],
                    )
                )
                code += self.generate_statement(stmt, indent)
        finally:
            self.current_unused_local_declaration_names = previous_unused_declarations
        return code

    def unused_local_declaration_names_for_statement(self, stmt, remaining_statements):
        name = getattr(stmt, "name", None)
        if not name or not isinstance(stmt, (VariableNode, ArrayNode)):
            return set()
        if self.statement_sequence_uses_identifier(remaining_statements, name):
            return set()
        return {name}

    def statement_sequence_uses_identifier(self, statements, name):
        for stmt in statements:
            if (
                isinstance(stmt, (VariableNode, ArrayNode))
                and getattr(stmt, "name", None) == name
            ):
                return False
            if self.statement_uses_identifier(stmt, name):
                return True
        return False

    def statement_uses_identifier(self, stmt, name):
        if isinstance(stmt, VariableNode):
            return self.node_uses_identifier(getattr(stmt, "initial_value", None), name)
        if isinstance(stmt, ArrayNode):
            return False
        return self.node_uses_identifier(stmt, name)

    def node_uses_identifier(self, node, name):
        if node is None:
            return False
        if isinstance(node, str):
            return re.search(rf"\b{re.escape(name)}\b", node) is not None
        if isinstance(node, IdentifierNode):
            return getattr(node, "name", None) == name
        if isinstance(node, VariableNode):
            if (
                getattr(node, "initial_value", None) is not None
                or getattr(node, "var_type", None) is not None
                or getattr(node, "vtype", None) is not None
            ):
                return self.node_uses_identifier(
                    getattr(node, "initial_value", None),
                    name,
                )
            return getattr(node, "name", None) == name
        if isinstance(node, ArrayNode):
            return False
        if isinstance(node, dict):
            return any(
                self.node_uses_identifier(value, name) for value in node.values()
            )
        if isinstance(node, (list, tuple, set)):
            return any(self.node_uses_identifier(value, name) for value in node)
        if hasattr(node, "__dict__"):
            for key, value in vars(node).items():
                if key in {"name", "type_name", "var_type", "vtype", "parent"}:
                    continue
                if key in {"body", "if_body", "else_body", "statements"}:
                    if self.statement_body_uses_identifier(value, name):
                        return True
                    continue
                if self.node_uses_identifier(value, name):
                    return True
            return False
        return False

    def statement_body_uses_identifier(self, body, name):
        if hasattr(body, "statements"):
            return self.statement_sequence_uses_identifier(body.statements, name)
        if isinstance(body, list):
            return self.statement_sequence_uses_identifier(body, name)
        return self.node_uses_identifier(body, name)

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, str):
            return init
        if isinstance(init, VariableNode) or (
            hasattr(init, "__class__") and "ExpressionStatement" in str(init.__class__)
        ):
            initializer = self.generate_statement(init, 0).strip().rstrip(";")
            return self.strip_unused_metal_declaration_attribute(initializer)
        return self.generate_expression(init).strip().rstrip(";")

    def generate_expression(self, expr):
        """Render a CrossGL AST expression into Metal expression syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            builtin_name = self.metal_builtin_expression_name(expr)
            if builtin_name != expr:
                return builtin_name
            if expr.rsplit("::", 1)[-1] == "None":
                option_default = self.option_none_default_expression()
                if option_default is not None:
                    return option_default
            if expr in self.local_variable_types:
                return self.metal_local_identifier_name(expr)
            return expr
        elif isinstance(expr, IdentifierNode):
            name = expr.name
            builtin_name = self.metal_builtin_expression_name(name)
            if builtin_name != name:
                return builtin_name
            if isinstance(name, str) and name.rsplit("::", 1)[-1] == "None":
                option_default = self.option_none_default_expression()
                if option_default is not None:
                    return option_default
            buffer_block_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if buffer_block_value is not None:
                return buffer_block_value
            if name in self.METAL_RAY_FLAG_VALUES:
                return f"{self.METAL_RAY_FLAG_VALUES[name]}u"
            if name in getattr(self, "enum_variant_constants", {}):
                return enum_value_expression(self, name)
            if (
                name not in self.local_variable_types
                and name in self.ambiguous_cbuffer_members
            ):
                raise ValueError(
                    f"Ambiguous cbuffer member reference '{name}' appears in multiple cbuffers"
                )
            if (
                name not in self.local_variable_types
                and name in self.cbuffer_member_references
            ):
                return self.cbuffer_member_references[name]
            return self.metal_local_identifier_name(name)
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, (VariableNode, BackendVariableNode)):
            # Fix infinite recursion - directly return the name
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            unsupported_value = (
                self.unsupported_metal_acceleration_structure_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            unsupported_value = (
                self.unsupported_metal_ray_function_table_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            if hasattr(expr, "name"):
                builtin_name = self.metal_builtin_expression_name(expr.name)
                if builtin_name != expr.name:
                    return builtin_name
                if (
                    isinstance(expr.name, str)
                    and expr.name.rsplit("::", 1)[-1] == "None"
                ):
                    option_default = self.option_none_default_expression()
                    if option_default is not None:
                        return option_default
                if expr.name in self.METAL_RAY_FLAG_VALUES:
                    return f"{self.METAL_RAY_FLAG_VALUES[expr.name]}u"
                if expr.name in self.local_variable_types:
                    return self.metal_local_identifier_name(expr.name)
                return enum_value_expression(self, expr.name)
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            operator = self.map_operator(expr.op)
            left = self.generate_binary_operand(expr.left, operator)
            right = self.generate_binary_operand(expr.right, operator, True)
            if operator in {"+", "-", "*", "/", "%"}:
                expected_type = self.current_expression_expected_type
                expression_width = self.vector_value_width(
                    self.expression_result_type(expr)
                )
                if (
                    expression_width is not None
                    and expression_width == self.vector_value_width(expected_type)
                ):
                    left = self.cast_binary_operand_to_expected_vector(
                        left, self.expression_result_type(expr.left), expected_type
                    )
                    right = self.cast_binary_operand_to_expected_vector(
                        right, self.expression_result_type(expr.right), expected_type
                    )
            return f"{left} {operator} {right}"
        elif isinstance(expr, (AssignmentNode, BackendAssignmentNode)):
            return self.generate_assignment(expr)
        elif isinstance(expr, ArrayLiteralNode):
            elements = ", ".join(
                self.generate_expression_with_expected(element, None)
                for element in expr.elements
            )
            return f"{{{elements}}}"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_unary_operand(expr.operand)
            return f"{self.map_operator(expr.op)}{operand}"
        elif isinstance(expr, CooperativeMatrixOpNode):
            return self.generate_cooperative_matrix_operation(expr)
        elif isinstance(expr, WaveOpNode):
            return self.generate_metal_wave_op_expression(expr)
        elif isinstance(expr, RayTracingOpNode):
            return self.generate_ray_tracing_op_expression(expr)
        elif isinstance(expr, MeshOpNode):
            mesh_call = self.generate_mesh_op_expression(expr)
            if mesh_call is not None:
                return mesh_call
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, RayQueryOpNode):
            return self.generate_metal_ray_query_call(
                expr.operation,
                expr.query_expr,
                expr.arguments,
            )
        elif isinstance(expr, ArrayAccessNode):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            stage_io_access = self.generate_metal_stage_io_lowered_array_access(expr)
            if stage_io_access is not None:
                return stage_io_access
            unsupported_value = (
                self.unsupported_metal_acceleration_structure_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            unsupported_value = (
                self.unsupported_metal_ray_function_table_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_array_load(expr)
            if block_load is not None:
                return block_load
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, ConstructorNode):
            enum_constructor = generate_enum_constructor_expression(self, expr)
            if enum_constructor is not None:
                return enum_constructor
            constructor = generate_struct_constructor_expression(self, expr)
            if constructor is not None:
                return constructor
            constructor_type = getattr(expr, "constructor_type", None)
            if self.is_builtin_value_constructor_type(constructor_type):
                metal_type = self.map_type(constructor_type)
                matrix_resize = self.generate_metal_matrix_resize_constructor(
                    metal_type, getattr(expr, "arguments", [])
                )
                if matrix_resize is not None:
                    return matrix_resize
                args = ", ".join(
                    self.generate_expression_with_expected(arg, None)
                    for arg in getattr(expr, "arguments", [])
                )
                return f"{metal_type}({args})"
            return str(expr)
        elif isinstance(expr, FunctionCallNode):
            option_payload = self.option_constructor_payload_expression(expr)
            if option_payload is not None:
                return option_payload

            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = expr.name
            numeric_trait_call = generate_numeric_trait_method_call(
                self,
                func_expr,
                expr.args,
            )
            if numeric_trait_call is not None:
                return numeric_trait_call

            func_name = None
            if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)

            unsupported_table_call = (
                self.unsupported_metal_ray_function_table_array_member_call(func_expr)
            )
            if unsupported_table_call is not None:
                return unsupported_table_call

            ray_query_operation = self.ray_query_operation_from_function_call(expr)
            if ray_query_operation is not None and isinstance(
                func_expr, MemberAccessNode
            ):
                return self.generate_metal_ray_query_call(
                    ray_query_operation,
                    getattr(
                        func_expr,
                        "object",
                        getattr(func_expr, "object_expr", None),
                    ),
                    expr.args,
                )

            static_generic_call = generate_static_generic_numeric_call(self, func_name)
            if static_generic_call is not None:
                return static_generic_call

            if func_name == "RayDesc" and self.requires_metal_builtin_ray_desc(
                func_name
            ):
                self.required_metal_ray_desc_runtime = True
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"CglRayDesc({args})"

            unsupported_call = self.unsupported_glsl_buffer_block_function_call(
                func_name
            )
            if unsupported_call is not None:
                return unsupported_call

            input_attachment_call = self.unsupported_input_attachment_call(func_name)
            if input_attachment_call is not None:
                return input_attachment_call

            enum_constructor = generate_enum_constructor_call(
                self, func_name, expr.args
            )
            if enum_constructor is not None:
                return enum_constructor

            glsl_block_atomic_call = self.generate_glsl_buffer_block_atomic_call(
                func_name, expr.args
            )
            if glsl_block_atomic_call is not None:
                return glsl_block_atomic_call

            buffer_atomic_call = self.generate_metal_buffer_resource_atomic_call(
                func_name, expr.args
            )
            if buffer_atomic_call is not None:
                return buffer_atomic_call

            synchronization_call = self.synchronization_function_call(
                func_name,
                expr.args,
                source_location=getattr(expr, "source_location", None),
            )
            if synchronization_call is not None:
                return synchronization_call

            wave_call = self.generate_metal_wave_operation(func_name, expr.args)
            if wave_call is not None:
                return wave_call

            atomic_call = self.generate_atomic_function_call(func_name, expr.args)
            if atomic_call is not None:
                return atomic_call

            mesh_output_call = self.generate_metal_mesh_output_call(
                func_name, expr.args
            )
            if mesh_output_call is not None:
                return mesh_output_call

            min_lod_clamp_call = self.generate_texture_min_lod_clamp_call(
                func_name, expr.args
            )
            if min_lod_clamp_call is not None:
                return min_lod_clamp_call

            texture_call = self.generate_texture_call(func_name, expr.args)
            if texture_call is not None:
                return texture_call

            buffer_call = self.generate_buffer_call(func_name, expr.args)
            if buffer_call is not None:
                return buffer_call
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                expr.args,
            )
            argument_func_name = func_name
            if specialized_func_name is not None:
                callee = specialized_func_name
                func_name = specialized_func_name
            if func_name == "normalize" and func_name not in self.user_function_names:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                call_name = (
                    "metal::normalize"
                    if self.metal_function_name_is_shadowed("normalize")
                    else "normalize"
                )
                return f"{call_name}({args})"
            if (
                func_name == "inverse"
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg_type = self.map_type(self.expression_result_type(expr.args[0]))
                if arg_type in {"float2x2", "float3x3", "float4x4"}:
                    self.required_metal_inverse_helpers.add(arg_type)
                    arg = self.generate_expression(expr.args[0])
                    return f"__crossgl_inverse_{arg_type}({arg})"
            if (
                func_name == "mod"
                and len(expr.args) == 2
                and func_name not in self.user_function_names
            ):
                left = self.generate_expression(expr.args[0])
                right = self.generate_expression(expr.args[1])
                return f"(({left}) - (({right}) * floor(({left}) / ({right}))))"
            if (
                func_name == "frac"
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg = self.generate_expression(expr.args[0])
                fract_name = (
                    "metal::fract"
                    if self.metal_function_name_is_shadowed("fract")
                    else "fract"
                )
                return f"{fract_name}({arg})"
            if (
                func_name == "lerp"
                and len(expr.args) == 3
                and func_name not in self.user_function_names
            ):
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                mix_name = (
                    "metal::mix"
                    if self.metal_function_name_is_shadowed("mix")
                    else "mix"
                )
                return f"{mix_name}({args})"
            if (
                func_name == "atan"
                and len(expr.args) == 2
                and func_name not in self.user_function_names
            ):
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                atan2_name = (
                    "metal::atan2"
                    if self.metal_function_name_is_shadowed("atan2")
                    else "atan2"
                )
                return f"{atan2_name}({args})"
            if (
                func_name in {"inverseSqrt", "inversesqrt"}
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg = self.generate_expression(expr.args[0])
                rsqrt_name = (
                    "metal::rsqrt"
                    if self.metal_function_name_is_shadowed("rsqrt")
                    else "rsqrt"
                )
                return f"{rsqrt_name}({arg})"
            derivative_name = self.METAL_DERIVATIVE_FUNCTION_ALIASES.get(func_name)
            if (
                derivative_name is not None
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg = self.generate_expression(expr.args[0])
                derivative_call_name = (
                    f"metal::{derivative_name}"
                    if self.metal_function_name_is_shadowed(derivative_name)
                    else derivative_name
                )
                return f"{derivative_call_name}({arg})"
            bitcast_call = self.generate_metal_bitcast_call(func_name, expr.args)
            if bitcast_call is not None:
                return bitcast_call
            integer_bit_call = self.generate_metal_integer_bit_call(
                func_name, expr.args
            )
            if integer_bit_call is not None:
                return integer_bit_call
            if (
                func_name in self.METAL_STDLIB_BUILTIN_FUNCTIONS
                and func_name not in self.user_function_names
            ):
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                call_name = (
                    f"metal::{func_name}"
                    if self.metal_function_name_is_shadowed(func_name)
                    else func_name
                )
                return f"{call_name}({args})"
            if func_name in [
                "float",
                "half",
                "float16",
                "min16float",
                "min10float",
                "double",
                "f16",
                "f32",
                "f64",
                "int",
                "i8",
                "i16",
                "i32",
                "char",
                "signed char",
                "int8",
                "int16",
                "int8_t",
                "int16_t",
                "int32_t",
                "int64",
                "int64_t",
                "long",
                "signed long",
                "ptrdiff_t",
                "min16int",
                "min12int",
                "uint",
                "u8",
                "u16",
                "u32",
                "uchar",
                "unsigned char",
                "uint8",
                "uint16",
                "uint8_t",
                "uint16_t",
                "uint32_t",
                "uint64",
                "uint64_t",
                "ulong",
                "unsigned long",
                "size_t",
                "min16uint",
                "short",
                "signed short",
                "ushort",
                "unsigned short",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "vec2<f16>",
                "vec3<f16>",
                "vec4<f16>",
                "vec2<f32>",
                "vec3<f32>",
                "vec4<f32>",
                "vec2<f64>",
                "vec3<f64>",
                "vec4<f64>",
                "vec2<i8>",
                "vec3<i8>",
                "vec4<i8>",
                "vec2<u8>",
                "vec3<u8>",
                "vec4<u8>",
                "vec2<i16>",
                "vec3<i16>",
                "vec4<i16>",
                "vec2<u16>",
                "vec3<u16>",
                "vec4<u16>",
                "vec2<i32>",
                "vec3<i32>",
                "vec4<i32>",
                "vec2<u32>",
                "vec3<u32>",
                "vec4<u32>",
                "vec2<bool>",
                "vec3<bool>",
                "vec4<bool>",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "packed_float2",
                "packed_float3",
                "packed_float4",
                "simd_float2",
                "simd_float3",
                "simd_float4",
                "half2",
                "half3",
                "half4",
                "packed_half2",
                "packed_half3",
                "packed_half4",
                "simd_half2",
                "simd_half3",
                "simd_half4",
                "f16vec2",
                "f16vec3",
                "f16vec4",
                "mat2",
                "mat3",
                "mat4",
                "mat2x2",
                "mat2x3",
                "mat2x4",
                "mat3x2",
                "mat3x3",
                "mat3x4",
                "mat4x2",
                "mat4x3",
                "mat4x4",
                "f16mat2",
                "f16mat3",
                "f16mat4",
                "f16mat2x2",
                "f16mat2x3",
                "f16mat2x4",
                "f16mat3x2",
                "f16mat3x3",
                "f16mat3x4",
                "f16mat4x2",
                "f16mat4x3",
                "f16mat4x4",
                "char2",
                "char3",
                "char4",
                "uchar2",
                "uchar3",
                "uchar4",
                "packed_int2",
                "packed_int3",
                "packed_int4",
                "packed_uint2",
                "packed_uint3",
                "packed_uint4",
                "simd_int2",
                "simd_int3",
                "simd_int4",
                "simd_uint2",
                "simd_uint3",
                "simd_uint4",
                "simd_float2x2",
                "simd_float2x3",
                "simd_float2x4",
                "simd_float3x2",
                "simd_float3x3",
                "simd_float3x4",
                "simd_float4x2",
                "simd_float4x3",
                "simd_float4x4",
                "simd_half2x2",
                "simd_half2x3",
                "simd_half2x4",
                "simd_half3x2",
                "simd_half3x3",
                "simd_half3x4",
                "simd_half4x2",
                "simd_half4x3",
                "simd_half4x4",
                "i8vec2",
                "i8vec3",
                "i8vec4",
                "u8vec2",
                "u8vec3",
                "u8vec4",
                "short2",
                "short3",
                "short4",
                "ushort2",
                "ushort3",
                "ushort4",
                "i16vec2",
                "i16vec3",
                "i16vec4",
                "u16vec2",
                "u16vec3",
                "u16vec4",
                "min16float2",
                "min16float3",
                "min16float4",
                "min10float2",
                "min10float3",
                "min10float4",
                "min16int2",
                "min16int3",
                "min16int4",
                "min12int2",
                "min12int3",
                "min12int4",
                "min16uint2",
                "min16uint3",
                "min16uint4",
                "min16float2x2",
                "min16float2x3",
                "min16float2x4",
                "min16float3x2",
                "min16float3x3",
                "min16float3x4",
                "min16float4x2",
                "min16float4x3",
                "min16float4x4",
                "min10float2x2",
                "min10float2x3",
                "min10float2x4",
                "min10float3x2",
                "min10float3x3",
                "min10float3x4",
                "min10float4x2",
                "min10float4x3",
                "min10float4x4",
            ]:
                metal_type = self.map_type(func_name)
                matrix_resize = self.generate_metal_matrix_resize_constructor(
                    metal_type, expr.args
                )
                if matrix_resize is not None:
                    return matrix_resize
                args = ", ".join(
                    self.generate_expression_with_expected(arg, None)
                    for arg in expr.args
                )
                return f"{metal_type}({args})"
            readonly_raw_buffer_call = self.readonly_raw_buffer_call_diagnostic(
                argument_func_name, expr.args
            )
            if readonly_raw_buffer_call is not None:
                return readonly_raw_buffer_call
            readonly_parameter_call = self.readonly_metal_parameter_call_diagnostic(
                argument_func_name, expr.args
            )
            if readonly_parameter_call is not None:
                return readonly_parameter_call
            mesh_context_call = self.metal_mesh_dispatch_context_call_diagnostic(
                func_name
            )
            if mesh_context_call is not None:
                return mesh_context_call
            address_space_call = self.address_space_call_diagnostic(
                argument_func_name, expr.args
            )
            if address_space_call is not None:
                return address_space_call
            wave_lane_call = self.metal_wave_lane_helper_call_diagnostic(func_name)
            if wave_lane_call is not None:
                return wave_lane_call
            readonly_mesh_payload_call = (
                self.readonly_metal_mesh_payload_call_diagnostic(
                    argument_func_name, expr.args
                )
            )
            if readonly_mesh_payload_call is not None:
                return readonly_mesh_payload_call
            self.validate_function_resource_argument_types(func_name, expr.args)
            self.validate_function_image_access_arguments(func_name, expr.args)
            args = self.generate_function_call_arguments(argument_func_name, expr.args)
            if self.function_call_matches_known_signature(func_name, expr.args):
                args.extend(
                    self.required_function_stage_output_argument_names(func_name)
                )
                args.extend(
                    self.required_function_stage_parameter_argument_names(func_name)
                )
                args.extend(
                    self.cbuffer_parameter_name(cbuffer)
                    for cbuffer in self.required_function_cbuffers(func_name)
                )
                args.extend(self.required_function_resource_argument_names(func_name))
                args.extend(
                    self.required_metal_mesh_dispatch_context_arguments(func_name)
                )
                wave_lane_args = self.required_metal_wave_lane_context_arguments(
                    func_name
                )
                if wave_lane_args is not None:
                    args.extend(wave_lane_args)
            args = ", ".join(args)
            return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            stage_io_access = self.generate_metal_stage_io_lowered_member_access(expr)
            if stage_io_access is not None:
                return stage_io_access
            unsupported_value = (
                self.unsupported_metal_ray_function_table_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_member_load(expr)
            if block_load is not None:
                return block_load
            suppress_suffix = self.member_access_is_image_load_component(expr)
            previous_suppression = self.suppress_image_load_component_suffix
            self.suppress_image_load_component_suffix = (
                previous_suppression or suppress_suffix
            )
            try:
                obj = self.generate_expression_with_expected(expr.object, None)
            finally:
                self.suppress_image_load_component_suffix = previous_suppression
            if self.member_access_uses_pointer_operator(expr):
                return f"{obj}->{expr.member}"
            return f"{obj}.{expr.member}"
        elif isinstance(expr, PointerAccessNode):
            pointer = self.generate_expression_with_expected(expr.pointer_expr, None)
            return f"{pointer}->{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition)} ? {self.generate_expression(expr.true_expr)} : {self.generate_expression(expr.false_expr)}"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            if hasattr(expr, "value"):
                value = expr.value
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                if (
                    literal_type == "uint"
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    return f"{value}u"
                if isinstance(value, bool):
                    return "true" if value else "false"
                if isinstance(value, str) and not (
                    value.startswith('"') and value.endswith('"')
                ):
                    return f'"{value}"'  # Add quotes for string literals
                return str(value)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", str(expr))
            if isinstance(name, str) and name.rsplit("::", 1)[-1] == "None":
                option_default = self.option_none_default_expression()
                if option_default is not None:
                    return option_default
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            if name in self.METAL_RAY_FLAG_VALUES:
                return f"{self.METAL_RAY_FLAG_VALUES[name]}u"
            if name in getattr(self, "enum_variant_constants", {}):
                return enum_value_expression(self, name)
            if (
                name not in self.local_variable_types
                and name in self.ambiguous_cbuffer_members
            ):
                raise ValueError(
                    f"Ambiguous cbuffer member reference '{name}' appears in multiple cbuffers"
                )
            if (
                name not in self.local_variable_types
                and name in self.cbuffer_member_references
            ):
                return self.cbuffer_member_references[name]
            return name
        else:
            return str(expr)

    def metal_function_name_is_shadowed(self, func_name):
        return (
            func_name in self.local_variable_types
            or func_name in self.user_function_names
        )

    def metal_bitcast_result_type(self, func_name, args):
        if (
            func_name not in self.METAL_BITCAST_FUNCTION_TARGETS
            or func_name in self.user_function_names
            or len(args or []) != 1
        ):
            return None

        argument_type = self.expression_result_type(args[0])
        mapped_argument_type = self.map_type(argument_type)
        match = re.fullmatch(r"(?:float|int|uint)([234])?", mapped_argument_type)
        if match is None:
            return self.METAL_BITCAST_FUNCTION_TARGETS[func_name]

        width = match.group(1) or ""
        return f"{self.METAL_BITCAST_FUNCTION_TARGETS[func_name]}{width}"

    def generate_metal_bitcast_call(self, func_name, args):
        target_type = self.metal_bitcast_result_type(func_name, args)
        if target_type is None:
            return None

        arg = self.generate_expression(args[0])
        return f"as_type<{target_type}>({arg})"

    def metal_integer_bit_result_type(self, func_name, args):
        if (
            func_name not in self.METAL_INTEGER_BIT_FUNCTION_TARGETS
            or func_name in self.user_function_names
            or len(args or []) != 1
        ):
            return None

        argument_type = self.expression_result_type(args[0])
        mapped_argument_type = self.map_type(argument_type)
        component_type = self.vector_component_type(mapped_argument_type)
        if component_type not in {"int", "uint"}:
            return None

        return mapped_argument_type

    def generate_metal_integer_bit_call(self, func_name, args):
        if self.metal_integer_bit_result_type(func_name, args) is None:
            return None

        intrinsic_name = self.METAL_INTEGER_BIT_FUNCTION_TARGETS[func_name]
        call_name = (
            f"metal::{intrinsic_name}"
            if self.metal_function_name_is_shadowed(intrinsic_name)
            else intrinsic_name
        )
        arg = self.generate_expression(args[0])
        return f"{call_name}({arg})"

    def binary_precedence(self, operator):
        return self.BINARY_PRECEDENCE.get(operator, 0)

    def binary_child_needs_parentheses(
        self, parent_operator, child, is_right_child=False
    ):
        if not isinstance(child, BinaryOpNode):
            return False

        parent_precedence = self.binary_precedence(parent_operator)
        child_operator = self.map_operator(getattr(child, "op", ""))
        child_precedence = self.binary_precedence(child_operator)
        if child_precedence < parent_precedence:
            return True
        if child_precedence > parent_precedence:
            return False
        return is_right_child and (
            parent_operator not in self.ASSOCIATIVE_BINARY_OPS
            or child_operator != parent_operator
        )

    def generate_binary_operand(self, expr, parent_operator=None, is_right_child=False):
        rendered = self.generate_expression(expr)
        if (
            parent_operator is not None
            and self.binary_child_needs_parentheses(
                parent_operator, expr, is_right_child
            )
        ) or isinstance(expr, TernaryOpNode):
            return f"({rendered})"
        return rendered

    def generate_unary_operand(self, expr):
        rendered = self.generate_expression(expr)
        if isinstance(expr, (BinaryOpNode, TernaryOpNode)):
            return f"({rendered})"
        return rendered

    def atomic_fence_operand_identifier(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None
        return str(name).lstrip(":").rsplit("::", 1)[-1]

    def atomic_fence_operand_text(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            return (
                f"{self.atomic_fence_operand_text(expr.left)} | "
                f"{self.atomic_fence_operand_text(expr.right)}"
            )
        name = self.atomic_fence_operand_identifier(expr)
        return name if name is not None else expression_debug_name(expr)

    def collect_atomic_fence_memory_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_atomic_fence_memory_flags(expr.left)
            right = self.collect_atomic_fence_memory_flags(expr.right)
            if left is None or right is None:
                return None
            return left + right
        name = self.atomic_fence_operand_identifier(expr)
        if name in self.METAL_ATOMIC_FENCE_MEMORY_FLAGS:
            return (name,)
        return None

    def metal_atomic_fence_contract_error(
        self,
        reason,
        args,
        *,
        source_location=None,
    ):
        rendered = [self.atomic_fence_operand_text(arg) for arg in args]
        requested_contract = ", ".join(rendered) or "<no operands>"
        capability = {
            "invalid-argument-count": "metal.atomic-thread-fence.contract",
            "unsupported-memory-flags": "metal.atomic-thread-fence.memory-flags",
            "unsupported-memory-order": "metal.atomic-thread-fence.memory-order",
            "unsupported-thread-scope": "metal.atomic-thread-fence.thread-scope",
        }[reason]
        return UnsupportedMetalFeatureError(
            "atomic-thread-fence-contract",
            "Metal codegen cannot represent atomicThreadFence("
            f"{requested_contract}) exactly: {reason.replace('-', ' ')}",
            missing_capabilities=(capability,),
            operation="atomicThreadFence",
            reason=reason,
            source_location=source_location,
        )

    def generate_metal_atomic_thread_fence(self, args, *, source_location=None):
        if len(args) != 3:
            raise self.metal_atomic_fence_contract_error(
                "invalid-argument-count",
                args,
                source_location=source_location,
            )

        flags = self.collect_atomic_fence_memory_flags(args[0])
        order = self.atomic_fence_operand_identifier(args[1])
        scope = self.atomic_fence_operand_identifier(args[2])
        if flags is None:
            raise self.metal_atomic_fence_contract_error(
                "unsupported-memory-flags",
                args,
                source_location=source_location,
            )
        if order not in self.METAL_ATOMIC_FENCE_MEMORY_ORDERS:
            raise self.metal_atomic_fence_contract_error(
                "unsupported-memory-order",
                args,
                source_location=source_location,
            )
        if scope not in self.METAL_ATOMIC_FENCE_THREAD_SCOPES:
            raise self.metal_atomic_fence_contract_error(
                "unsupported-thread-scope",
                args,
                source_location=source_location,
            )

        rendered_flags = " | ".join(f"metal::mem_flags::{flag}" for flag in flags)
        return (
            f"metal::atomic_thread_fence({rendered_flags}, "
            f"metal::{order}, metal::{scope})"
        )

    def synchronization_function_call(self, func_name, args, *, source_location=None):
        if func_name in self.user_function_names:
            return None
        if func_name == "atomicThreadFence":
            return self.generate_metal_atomic_thread_fence(
                args, source_location=source_location
            )
        if args:
            return None
        return {
            "barrier": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "workgroupBarrier": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "workgroupExecutionBarrier": "threadgroup_barrier(mem_flags::mem_none)",
            "groupMemoryBarrier": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "memoryBarrierShared": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "memoryBarrierBuffer": "threadgroup_barrier(mem_flags::mem_device)",
            "deviceMemoryBarrier": "threadgroup_barrier(mem_flags::mem_device)",
            "memoryBarrierImage": "threadgroup_barrier(mem_flags::mem_texture)",
            "memoryBarrier": "threadgroup_barrier(mem_flags::mem_device)",
            "allMemoryBarrier": (
                "threadgroup_barrier(mem_flags::mem_device | "
                "mem_flags::mem_threadgroup | mem_flags::mem_texture)"
            ),
        }.get(func_name)

    def generate_cooperative_matrix_operation(self, node):
        """Render a canonical cooperative-matrix operation with Metal semantics."""
        operation = node.operation
        arguments = list(node.arguments)
        exact_arities = {
            "element": 2,
            "multiply": 2,
            "multiply_accumulate": 4,
            "elementwise_add": 2,
            "elementwise_subtract": 2,
            "negate": 1,
        }
        expected_arity = exact_arities.get(operation)
        if expected_arity is not None and len(arguments) != expected_arity:
            raise UnsupportedMetalFeatureError(
                "cooperative-matrix-operation-arity",
                f"Metal cooperative-matrix operation '{operation}' expects "
                f"{expected_arity} arguments, got {len(arguments)}",
                operation=operation,
                reason="invalid-argument-count",
                source_location=getattr(node, "source_location", None),
            )
        if operation in {"load", "store"} and len(arguments) < 2:
            raise UnsupportedMetalFeatureError(
                "cooperative-matrix-operation-arity",
                f"Metal cooperative-matrix operation '{operation}' expects at "
                f"least 2 arguments, got {len(arguments)}",
                operation=operation,
                reason="invalid-argument-count",
                source_location=getattr(node, "source_location", None),
            )

        rendered = [self.generate_expression(argument) for argument in arguments]
        if operation == "element":
            return f"{rendered[0]}.thread_elements()[{rendered[1]}]"
        if operation == "multiply":
            return f"({rendered[0]} * {rendered[1]})"
        if operation == "elementwise_add":
            return f"({rendered[0]} + {rendered[1]})"
        if operation == "elementwise_subtract":
            return f"({rendered[0]} - {rendered[1]})"
        if operation == "negate":
            return f"(-{rendered[0]})"
        if operation == "store":
            rendered = [rendered[1], rendered[0], *rendered[2:]]
        intrinsic = self.METAL_COOPERATIVE_MATRIX_FUNCTIONS.get(operation)
        if intrinsic is not None:
            return f"{intrinsic}({', '.join(rendered)})"

        raise UnsupportedMetalFeatureError(
            "cooperative-matrix-operation-lowering",
            f"Metal codegen cannot lower cooperative-matrix operation "
            f"'{operation}' without changing its semantics",
            missing_capabilities=("metal.cooperative-matrix-operation-lowering",),
            operation=operation,
            reason="operation-not-representable",
            source_location=getattr(node, "source_location", None),
        )

    def generate_metal_wave_op_expression(self, node):
        return self.generate_metal_wave_operation(node.operation, node.arguments)

    def generate_metal_wave_operation(self, operation, arguments):
        expected_arity = self.METAL_WAVE_INTRINSIC_ARITIES.get(operation)
        if expected_arity is None:
            return None

        actual_arity = len(arguments)
        if actual_arity != expected_arity:
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                f"expects {expected_arity} arguments, got {actual_arity}",
            )

        unsupported_reason = self.METAL_WAVE_UNSUPPORTED_OPERATIONS.get(operation)
        if unsupported_reason is not None:
            return self.metal_wave_diagnostic_expression(
                operation, arguments, unsupported_reason
            )

        type_diagnostic = self.metal_wave_type_diagnostic(operation, arguments)
        if type_diagnostic is not None:
            return type_diagnostic

        if operation == "WaveGetLaneIndex":
            lane_index_parameter = self.current_metal_wave_lane_index_parameter
            if lane_index_parameter is not None:
                return lane_index_parameter
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                "requires a compute-stage thread_index_in_simdgroup value",
            )
        if operation == "WaveGetLaneCount":
            lane_count_parameter = self.current_metal_wave_lane_count_parameter
            if lane_count_parameter is not None:
                return lane_count_parameter
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                "requires a compute-stage threads_per_simdgroup value",
            )
        if operation == "WaveIsFirstLane":
            return "simd_is_first()"
        if operation == "WaveActiveCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"uint(popcount(simd_vote::vote_t(simd_ballot({predicate}))))"
        if operation == "WavePrefixCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"simd_prefix_exclusive_sum(({predicate}) ? 1u : 0u)"
        if operation == "WaveActiveBallot":
            self.required_metal_wave_ballot_helper = True
            predicate = self.generate_expression(arguments[0])
            return f"__crossgl_metal_wave_ballot({predicate})"
        if operation == "WaveActiveAllEqual":
            value = self.generate_expression(arguments[0])
            mapped_type, component_type, _array_suffix = (
                self.metal_wave_argument_mapped_type(arguments[0])
            )
            equality = f"{value} == simd_broadcast_first({value})"
            if (
                mapped_type is not None
                and component_type is not None
                and mapped_type != component_type
            ):
                equality = f"all({equality})"
            return f"simd_all({equality})"
        if operation == "WaveMatch":
            lane_count_parameter = self.current_metal_wave_lane_count_parameter
            if lane_count_parameter is None:
                return self.metal_wave_diagnostic_expression(
                    operation,
                    arguments,
                    "requires a compute-stage threads_per_simdgroup value",
                )
            self.required_metal_wave_match_helper = True
            value = self.generate_expression(arguments[0])
            return f"__crossgl_metal_wave_match({value}, {lane_count_parameter})"
        if operation in self.METAL_WAVE_MULTI_PREFIX_INTRINSICS:
            lane_index_parameter = self.current_metal_wave_lane_index_parameter
            lane_count_parameter = self.current_metal_wave_lane_count_parameter
            if lane_index_parameter is None or lane_count_parameter is None:
                return self.metal_wave_diagnostic_expression(
                    operation,
                    arguments,
                    (
                        "requires compute-stage thread_index_in_simdgroup "
                        "and threads_per_simdgroup values"
                    ),
                )
            self.required_metal_wave_ballot_helper = True
            self.required_metal_wave_mask_contains_helper = True
            self.required_metal_wave_multi_prefix_helpers.add(operation)
            value = self.generate_expression(arguments[0])
            mask = self.generate_expression(arguments[1])
            helper = self.METAL_WAVE_MULTI_PREFIX_HELPERS[operation]
            return (
                f"{helper}({value}, {mask}, {lane_index_parameter}, "
                f"{lane_count_parameter})"
            )
        if operation == "WaveReadLaneAt":
            value = self.generate_expression(arguments[0])
            lane = self.generate_expression(arguments[1])
            return f"simd_broadcast({value}, ushort({lane}))"
        if operation in self.METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS:
            value = self.generate_expression(arguments[0])
            fill = self.generate_expression(arguments[1])
            delta = self.generate_expression(arguments[2])
            mapped_type, component_type, _array_suffix = (
                self.metal_wave_argument_mapped_type(arguments[0])
            )
            boolean_payload = (
                mapped_type == "bool"
                or component_type == "bool"
                or all(
                    self.metal_wave_expression_is_boolean(argument)
                    for argument in arguments[:2]
                )
            )
            if boolean_payload:
                component_count = self.value_component_count(mapped_type) or 1
                payload_type = (
                    "uint" if component_count == 1 else f"uint{component_count}"
                )
                zero = "0u" if component_count == 1 else f"{payload_type}(0u)"
                return (
                    f"(simd_shuffle_and_fill_up({payload_type}({value}), "
                    f"{payload_type}({fill}), ushort({delta})) != {zero})"
                )
            result = f"simd_shuffle_and_fill_up({value}, {fill}, ushort({delta}))"
            expected_type = self.map_type(self.current_expression_expected_type)
            if expected_type == "bool":
                return (
                    f"({result} != "
                    f"{self.diagnostic_zero_value_for_type(mapped_type)})"
                )
            return result
        if operation == "QuadReadLaneAt":
            value = self.generate_expression(arguments[0])
            lane = self.generate_expression(arguments[1])
            return f"quad_broadcast({value}, ushort({lane}))"
        if operation == "QuadReadAcrossX":
            value = self.generate_expression(arguments[0])
            return f"quad_shuffle_xor({value}, ushort(1))"
        if operation == "QuadReadAcrossY":
            value = self.generate_expression(arguments[0])
            return f"quad_shuffle_xor({value}, ushort(2))"
        if operation == "QuadReadAcrossDiagonal":
            value = self.generate_expression(arguments[0])
            return f"quad_shuffle_xor({value}, ushort(3))"

        mapped = self.METAL_WAVE_DIRECT_MAPPINGS.get(operation)
        if mapped is None:
            return self.metal_wave_diagnostic_expression(
                operation, arguments, "is not recognized by the Metal backend"
            )
        args = ", ".join(self.generate_expression(arg) for arg in arguments)
        return f"{mapped}({args})"

    def metal_wave_result_type(self, operation, arguments):
        if operation in self.METAL_WAVE_UINT_RESULT_INTRINSICS:
            return "uint"
        if operation in self.METAL_WAVE_BOOL_RESULT_INTRINSICS:
            return "bool"
        if operation in self.METAL_WAVE_UINT4_RESULT_INTRINSICS:
            return "uint4"
        if operation in self.METAL_WAVE_VALUE_RESULT_INTRINSICS and arguments:
            return self.expression_result_type(arguments[0])
        return None

    def metal_wave_argument_mapped_type(self, argument):
        literal_type = getattr(getattr(argument, "literal_type", None), "name", None)
        argument_type = literal_type or self.expression_result_type(argument)
        if argument_type is None:
            return None, None, None
        mapped_type = self.map_type(argument_type)
        component_type = self.vector_component_type(mapped_type)
        return mapped_type, component_type, split_array_type_suffix(mapped_type)[1]

    def metal_wave_validate_predicate_argument(self, operation, argument):
        mapped_type, _component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if array_suffix or mapped_type != "bool":
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                ("predicate argument must be scalar bool, got " f"{mapped_type}"),
            )
        return None

    def metal_wave_validate_value_argument(
        self, operation, argument, allowed_components, description
    ):
        mapped_type, component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if array_suffix or component_type not in allowed_components:
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"value argument must be {description}, got {mapped_type}",
            )
        return None

    def metal_wave_validate_lane_argument(self, operation, argument, role):
        mapped_type, _component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if array_suffix or mapped_type not in {"int", "uint"}:
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"{role} must be scalar int or uint, got {mapped_type}",
            )
        return None

    def metal_wave_validate_match_argument(self, operation, argument):
        mapped_type, component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if (
            array_suffix
            or mapped_type != component_type
            or component_type not in self.METAL_WAVE_NUMERIC_COMPONENT_TYPES
        ):
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"value argument must be numeric scalar, got {mapped_type}",
            )
        return None

    def metal_wave_mapped_type_is_matrix(self, mapped_type):
        return self.is_matrix_value_type(mapped_type)

    def metal_wave_validate_non_matrix_value_argument(
        self, operation, argument, allowed_components, description
    ):
        mapped_type, component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if (
            array_suffix
            or self.metal_wave_mapped_type_is_matrix(mapped_type)
            or component_type not in allowed_components
        ):
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"value argument must be {description}, got {mapped_type}",
            )
        return None

    def metal_wave_validate_multi_prefix_mask_argument(
        self, operation, argument, value_argument=None
    ):
        mapped_type, _component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        diagnostic_arguments = [value_argument] if value_argument is not None else []
        diagnostic_arguments.append(argument)
        if array_suffix or mapped_type != "uint4":
            return self.metal_wave_diagnostic_expression(
                operation,
                diagnostic_arguments,
                f"mask argument must be uint4, got {mapped_type}",
            )
        return None

    def metal_wave_validate_quad_lane_range(self, operation, argument):
        lane_index = self.literal_int_value(argument, self.literal_int_constants)
        if lane_index is None:
            return None
        if not 0 <= lane_index <= 3:
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"quad lane index must be in the range 0 to 3, got {lane_index}",
            )
        return None

    def metal_wave_validate_shuffle_and_fill_arguments(self, operation, arguments):
        allowed_components = self.METAL_WAVE_NUMERIC_COMPONENT_TYPES | {"bool"}
        mapped_types = []
        for argument, role in ((arguments[0], "value"), (arguments[1], "fill")):
            mapped_type, component_type, array_suffix = (
                self.metal_wave_argument_mapped_type(argument)
            )
            mapped_types.append(mapped_type)
            if mapped_type is None:
                continue
            if self.is_scalar_value_type(mapped_type):
                component_type = mapped_type
            if (
                array_suffix
                or self.metal_wave_mapped_type_is_matrix(mapped_type)
                or component_type not in allowed_components
            ):
                return self.metal_wave_diagnostic_expression(
                    operation,
                    arguments,
                    f"{role} argument must be a basic scalar or vector, got "
                    f"{mapped_type}",
                )

        value_type, fill_type = mapped_types
        if value_type is not None and fill_type is not None and value_type != fill_type:
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                "value and fill arguments must have matching types, got "
                f"{value_type} and {fill_type}",
            )
        return self.metal_wave_validate_lane_argument(operation, arguments[2], "delta")

    def metal_wave_expression_is_boolean(self, expression):
        expression_type = self.expression_result_type(expression)
        mapped_type = self.map_type(expression_type) if expression_type else None
        if mapped_type == "bool" or self.vector_component_type(mapped_type) == "bool":
            return True
        literal_type = getattr(getattr(expression, "literal_type", None), "name", None)
        return self.map_type(literal_type) == "bool" if literal_type else False

    def metal_wave_type_diagnostic(self, operation, arguments):
        if operation in self.METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS:
            return self.metal_wave_validate_shuffle_and_fill_arguments(
                operation, arguments
            )
        if operation == "WaveMultiPrefixCountBits":
            diagnostic = self.metal_wave_validate_predicate_argument(
                operation, arguments[0]
            )
            if diagnostic is not None:
                return diagnostic
            return self.metal_wave_validate_multi_prefix_mask_argument(
                operation, arguments[1], arguments[0]
            )
        if operation in self.METAL_WAVE_MULTI_PREFIX_NUMERIC_INTRINSICS:
            diagnostic = self.metal_wave_validate_non_matrix_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric scalar or vector",
            )
            if diagnostic is not None:
                return diagnostic
            return self.metal_wave_validate_multi_prefix_mask_argument(
                operation, arguments[1], arguments[0]
            )
        if operation in self.METAL_WAVE_MULTI_PREFIX_INTEGER_INTRINSICS:
            diagnostic = self.metal_wave_validate_non_matrix_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_INTEGER_COMPONENT_TYPES,
                "integer scalar or vector",
            )
            if diagnostic is not None:
                return diagnostic
            return self.metal_wave_validate_multi_prefix_mask_argument(
                operation, arguments[1], arguments[0]
            )
        if operation in self.METAL_WAVE_BOOL_ARGUMENT_INTRINSICS:
            return self.metal_wave_validate_predicate_argument(operation, arguments[0])
        if operation in self.METAL_WAVE_NUMERIC_VALUE_INTRINSICS:
            return self.metal_wave_validate_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric scalar or vector",
            )
        if operation in self.METAL_WAVE_INTEGER_VALUE_INTRINSICS:
            return self.metal_wave_validate_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_INTEGER_COMPONENT_TYPES,
                "integer scalar or vector",
            )
        if operation == "WaveActiveAllEqual":
            return self.metal_wave_validate_non_matrix_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric scalar or vector",
            )
        if operation in self.METAL_WAVE_SIMDGROUP_VALUE_INTRINSICS:
            diagnostic = self.metal_wave_validate_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric or integer scalar or vector",
            )
            if diagnostic is not None:
                return diagnostic
        if operation == "WaveMatch":
            diagnostic = self.metal_wave_validate_match_argument(
                operation, arguments[0]
            )
            if diagnostic is not None:
                return diagnostic

        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            diagnostic = self.metal_wave_validate_lane_argument(
                operation, arguments[1], "lane index"
            )
            if diagnostic is not None:
                return diagnostic
            if operation == "QuadReadLaneAt":
                return self.metal_wave_validate_quad_lane_range(operation, arguments[1])
        return None

    def metal_wave_default_value(self, operation, arguments):
        result_type = (
            self.current_expression_expected_type
            or self.metal_wave_result_type(operation, arguments)
        )
        if result_type:
            return self.diagnostic_zero_value_for_type(result_type)
        if operation in self.METAL_WAVE_BOOL_RESULT_INTRINSICS:
            return "false"
        if operation in self.METAL_WAVE_UINT4_RESULT_INTRINSICS:
            return "uint4(0)"
        return "0u"

    def metal_wave_diagnostic_expression(self, operation, arguments, reason):
        return (
            f"/* unsupported Metal wave intrinsic: {operation} {reason} */ "
            f"{self.metal_wave_default_value(operation, arguments)}"
        )

    def generate_metal_ray_query_call(self, operation, query_expr, arguments):
        self.require_metal_ray_query_runtime()
        self.required_metal_ray_query_helpers.add(str(operation))
        helper_name = self.metal_ray_query_helper_name(operation)
        args = [self.generate_expression(query_expr)]
        args.extend(self.generate_expression(arg) for arg in arguments or [])
        return f"{helper_name}({', '.join(args)})"

    def metal_ray_query_default_value(self, return_type):
        if return_type == "bool":
            return "false"
        if return_type == "float":
            return "0.0"
        if return_type == "float2":
            return "float2(0.0)"
        if return_type == "float3":
            return "float3(0.0)"
        if return_type == "float3x4":
            return "float3x4(0.0)"
        return "0u"

    def generate_metal_ray_query_helpers(self):
        if not (
            self.required_metal_ray_query_runtime
            or self.required_metal_ray_desc_runtime
            or self.required_metal_ray_query_helpers
        ):
            return ""

        code = ""
        if self.required_metal_ray_desc_runtime:
            code += (
                "struct CglRayDesc {\n"
                "    float3 origin;\n"
                "    float t_min;\n"
                "    float3 direction;\n"
                "    float t_max;\n"
                "\n"
                "    CglRayDesc() {}\n"
                "\n"
                "    CglRayDesc(float3 origin_value, float t_min_value, "
                "float3 direction_value, float t_max_value)\n"
                "        : origin(origin_value),\n"
                "          t_min(t_min_value),\n"
                "          direction(direction_value),\n"
                "          t_max(t_max_value) {}\n"
                "};\n\n"
            )
        if (
            self.required_metal_ray_query_runtime
            or self.required_metal_ray_query_helpers
        ):
            code += "struct CglRayQuery {\n" "    uint state;\n" "};\n\n"

        for operation in sorted(self.required_metal_ray_query_helpers):
            return_type = self.metal_ray_query_method_return_type(operation)
            if return_type is None:
                continue
            helper_name = self.metal_ray_query_helper_name(operation)
            if operation == "TraceRayInline":
                code += (
                    "template <typename Accel, typename Flags, typename Mask, "
                    "typename Ray>\n"
                    f"inline void {helper_name}(\n"
                    "    CglRayQuery query,\n"
                    "    Accel acceleration_structure,\n"
                    "    Flags ray_flags,\n"
                    "    Mask instance_mask,\n"
                    "    Ray ray) {\n"
                    "    (void)query;\n"
                    "    (void)acceleration_structure;\n"
                    "    (void)ray_flags;\n"
                    "    (void)instance_mask;\n"
                    "    (void)ray;\n"
                    "}\n\n"
                    "template <typename Accel, typename Flags, typename Mask, "
                    "typename Origin, typename MinDistance, typename Direction, "
                    "typename MaxDistance>\n"
                    f"inline void {helper_name}(\n"
                    "    CglRayQuery query,\n"
                    "    Accel acceleration_structure,\n"
                    "    Flags ray_flags,\n"
                    "    Mask instance_mask,\n"
                    "    Origin origin,\n"
                    "    MinDistance min_distance,\n"
                    "    Direction direction,\n"
                    "    MaxDistance max_distance) {\n"
                    "    (void)query;\n"
                    "    (void)acceleration_structure;\n"
                    "    (void)ray_flags;\n"
                    "    (void)instance_mask;\n"
                    "    (void)origin;\n"
                    "    (void)min_distance;\n"
                    "    (void)direction;\n"
                    "    (void)max_distance;\n"
                    "}\n\n"
                )
                continue
            if operation in {
                "GenerateIntersection",
                "CommitProceduralPrimitiveHit",
                "CandidateTriangleVertexPositions",
                "CommittedTriangleVertexPositions",
            }:
                code += (
                    "template <typename Arg>\n"
                    f"inline void {helper_name}(CglRayQuery query, "
                    "const Arg& arg) {\n"
                    "    (void)query;\n"
                    "    (void)arg;\n"
                    "}\n\n"
                )
                continue
            if return_type == "void":
                code += (
                    f"inline void {helper_name}(CglRayQuery query) {{\n"
                    "    (void)query;\n"
                    "}\n\n"
                )
                continue
            code += (
                f"inline {return_type} {helper_name}(CglRayQuery query) {{\n"
                "    (void)query;\n"
                f"    return {self.metal_ray_query_default_value(return_type)};\n"
                "}\n\n"
            )
        return code

    def generate_metal_wave_helpers(self):
        code = ""
        if self.required_metal_wave_ballot_helper:
            code += (
                "uint4 __crossgl_metal_wave_ballot(bool predicate) {\n"
                "    simd_vote::vote_t mask = simd_vote::vote_t(simd_ballot(predicate));\n"
                "    return uint4(\n"
                "        uint(mask & simd_vote::vote_t(0xffffffffu)),\n"
                "        uint((mask >> simd_vote::vote_t(32u)) & "
                "simd_vote::vote_t(0xffffffffu)),\n"
                "        0u,\n"
                "        0u);\n"
                "}\n\n"
            )
        if self.required_metal_wave_match_helper:
            code += (
                "template <typename T>\n"
                "uint4 __crossgl_metal_wave_match(T value, uint laneCount) {\n"
                "    uint4 mask = uint4(0u);\n"
                "    for (uint lane = 0u; lane < laneCount; ++lane) {\n"
                "        if (simd_broadcast(value, ushort(lane)) == value) {\n"
                "            if (lane < 32u) {\n"
                "                mask.x |= (1u << lane);\n"
                "            } else if (lane < 64u) {\n"
                "                mask.y |= (1u << (lane - 32u));\n"
                "            } else if (lane < 96u) {\n"
                "                mask.z |= (1u << (lane - 64u));\n"
                "            } else {\n"
                "                mask.w |= (1u << (lane - 96u));\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    return mask;\n"
                "}\n\n"
            )
        if self.required_metal_wave_mask_contains_helper:
            code += (
                "bool __crossgl_metal_wave_mask_contains(uint4 mask, uint lane) {\n"
                "    if (lane < 32u) {\n"
                "        return (mask.x & (1u << lane)) != 0u;\n"
                "    }\n"
                "    if (lane < 64u) {\n"
                "        return (mask.y & (1u << (lane - 32u))) != 0u;\n"
                "    }\n"
                "    if (lane < 96u) {\n"
                "        return (mask.z & (1u << (lane - 64u))) != 0u;\n"
                "    }\n"
                "    return (mask.w & (1u << (lane - 96u))) != 0u;\n"
                "}\n\n"
            )
        for operation in self.METAL_WAVE_MULTI_PREFIX_HELPERS:
            if operation not in self.required_metal_wave_multi_prefix_helpers:
                continue
            helper_name = self.METAL_WAVE_MULTI_PREFIX_HELPERS[operation]
            if operation == "WaveMultiPrefixCountBits":
                code += (
                    f"uint {helper_name}(bool value, uint4 mask, uint laneIndex, "
                    "uint laneCount) {\n"
                    "    uint laneValue = value ? 1u : 0u;\n"
                    "    uint result = 0u;\n"
                    "    uint4 activeMask = __crossgl_metal_wave_ballot(true);\n"
                    "    uint limit = min(laneIndex, laneCount);\n"
                    "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                    "        if (__crossgl_metal_wave_mask_contains(mask, lane) && "
                    "__crossgl_metal_wave_mask_contains(activeMask, lane)) {\n"
                    "            result += simd_broadcast(laneValue, ushort(lane));\n"
                    "        }\n"
                    "    }\n"
                    "    return result;\n"
                    "}\n\n"
                )
                continue
            if operation == "WaveMultiPrefixSum":
                identity = "T(0)"
                assignment = "+="
            elif operation == "WaveMultiPrefixProduct":
                identity = "T(1)"
                assignment = "*="
            elif operation == "WaveMultiPrefixBitAnd":
                identity = "~T(0)"
                assignment = "&="
            elif operation == "WaveMultiPrefixBitOr":
                identity = "T(0)"
                assignment = "|="
            else:
                identity = "T(0)"
                assignment = "^="
            code += (
                "template <typename T>\n"
                f"T {helper_name}(T value, uint4 mask, uint laneIndex, "
                "uint laneCount) {\n"
                f"    T result = {identity};\n"
                "    uint4 activeMask = __crossgl_metal_wave_ballot(true);\n"
                "    uint limit = min(laneIndex, laneCount);\n"
                "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                "        if (__crossgl_metal_wave_mask_contains(mask, lane) && "
                "__crossgl_metal_wave_mask_contains(activeMask, lane)) {\n"
                f"            result {assignment} simd_broadcast(value, ushort(lane));\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def generate_metal_inverse_helpers(self):
        code = ""
        required = getattr(self, "required_metal_inverse_helpers", set())
        if "float2x2" in required:
            code += (
                "static inline float2x2 __crossgl_inverse_float2x2(float2x2 m) {\n"
                "    float det = m[0][0] * m[1][1] - m[1][0] * m[0][1];\n"
                "    if (abs(det) <= 1.0e-8) {\n"
                "        return float2x2(1.0);\n"
                "    }\n"
                "    return float2x2(\n"
                "        float2(m[1][1], -m[0][1]),\n"
                "        float2(-m[1][0], m[0][0])) / det;\n"
                "}\n\n"
            )
        if "float3x3" in required:
            code += (
                "static inline float3x3 __crossgl_inverse_float3x3(float3x3 m) {\n"
                "    float3 c0 = m[0];\n"
                "    float3 c1 = m[1];\n"
                "    float3 c2 = m[2];\n"
                "    float3 r0 = cross(c1, c2);\n"
                "    float3 r1 = cross(c2, c0);\n"
                "    float3 r2 = cross(c0, c1);\n"
                "    float det = dot(c0, r0);\n"
                "    if (abs(det) <= 1.0e-8) {\n"
                "        return float3x3(1.0);\n"
                "    }\n"
                "    return transpose(float3x3(r0, r1, r2)) / det;\n"
                "}\n\n"
            )
        if "float4x4" in required:
            code += (
                "static inline float __crossgl_det3_float4x4(\n"
                "    float a00, float a01, float a02,\n"
                "    float a10, float a11, float a12,\n"
                "    float a20, float a21, float a22) {\n"
                "    return a00 * (a11 * a22 - a12 * a21)\n"
                "         - a01 * (a10 * a22 - a12 * a20)\n"
                "         + a02 * (a10 * a21 - a11 * a20);\n"
                "}\n\n"
                "static inline float __crossgl_cofactor_float4x4(\n"
                "    float4x4 m, int column, int row) {\n"
                "    float values[9];\n"
                "    int index = 0;\n"
                "    for (int c = 0; c < 4; ++c) {\n"
                "        if (c == column) {\n"
                "            continue;\n"
                "        }\n"
                "        for (int r = 0; r < 4; ++r) {\n"
                "            if (r == row) {\n"
                "                continue;\n"
                "            }\n"
                "            values[index++] = m[c][r];\n"
                "        }\n"
                "    }\n"
                "    float minor_det = __crossgl_det3_float4x4(\n"
                "        values[0], values[1], values[2],\n"
                "        values[3], values[4], values[5],\n"
                "        values[6], values[7], values[8]);\n"
                "    return ((column + row) & 1) ? -minor_det : minor_det;\n"
                "}\n\n"
                "static inline float4x4 __crossgl_inverse_float4x4(float4x4 m) {\n"
                "    float det = determinant(m);\n"
                "    if (abs(det) <= 1.0e-8) {\n"
                "        return float4x4(1.0);\n"
                "    }\n"
                "    float4x4 result;\n"
                "    for (int c = 0; c < 4; ++c) {\n"
                "        for (int r = 0; r < 4; ++r) {\n"
                "            result[c][r] = __crossgl_cofactor_float4x4(m, r, c) / det;\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def generate_ray_tracing_op_expression(self, expr):
        raw_args = self.normalized_metal_intrinsic_args(expr.arguments)
        rendered_args = [self.generate_expression(arg) for arg in raw_args]
        if expr.operation == "TraceRay":
            trace_ray = self.generate_metal_trace_ray(raw_args, rendered_args)
            if trace_ray is not None:
                return trace_ray
            return self.unsupported_metal_ray_tracing_intrinsic(
                "TraceRay",
                "expected acceleration structure, flags, mask, SBT offsets, "
                "origin, tmin, direction, tmax, and payload location",
            )

        if expr.operation == "CallShader":
            call_shader = self.generate_metal_call_shader(raw_args, rendered_args)
            if call_shader is not None:
                return call_shader
            return self.unsupported_metal_ray_tracing_intrinsic(
                "CallShader",
                "requires one visible_function_table resource, or an explicit "
                "visible_function_table argument",
            )

        unsupported_reasons = {
            "ReportHit": (
                "intersection acceptance is expressed through the Metal intersection return value"
            ),
            "AcceptHitAndEndSearch": (
                "Metal hit acceptance is controlled by intersection results"
            ),
            "IgnoreHit": "Metal does not expose a direct ignore-intersection intrinsic",
        }
        reason = unsupported_reasons.get(expr.operation)
        if reason is not None:
            return self.unsupported_metal_ray_tracing_intrinsic(expr.operation, reason)

        return f"{expr.operation}({', '.join(rendered_args)})"

    def normalized_metal_intrinsic_args(self, args):
        normalized = []
        index = 0
        while index < len(args):
            arg = args[index]
            if self.is_unary_deref_marker(arg) and index + 1 < len(args):
                normalized.append(UnaryOpNode("*", args[index + 1]))
                index += 2
                continue
            normalized.append(arg)
            index += 1
        return normalized

    def is_unary_deref_marker(self, arg):
        return isinstance(arg, IdentifierNode) and getattr(arg, "name", None) == "*"

    def generate_metal_trace_ray(self, raw_args, rendered_args):
        if len(rendered_args) != 11:
            return None

        acceleration_structure_argument = (
            self.metal_trace_ray_acceleration_structure_argument(
                raw_args[0], rendered_args[0]
            )
        )
        if acceleration_structure_argument["reason"] is not None:
            return self.unsupported_metal_ray_tracing_intrinsic(
                "TraceRay acceleration structure",
                acceleration_structure_argument["reason"],
            )

        acceleration_structure = acceleration_structure_argument["argument"]
        instance_mask = rendered_args[2]
        origin = rendered_args[6]
        min_distance = rendered_args[7]
        direction = rendered_args[8]
        max_distance = rendered_args[9]
        acceleration_structure_type = acceleration_structure_argument["type"]
        intersection_function_table = self.default_intersection_function_table(
            acceleration_structure_type
        )
        intersector_tags = (
            intersection_function_table["tags"]
            if intersection_function_table is not None
            else self.default_metal_intersector_tags(acceleration_structure_type)
        )
        ray_name = self.next_metal_temp_variable("ray")
        intersector_name = self.next_metal_temp_variable("intersector")
        intersection_name = self.next_metal_temp_variable("intersection")
        intersect_args = [ray_name, acceleration_structure]
        if acceleration_structure_type != "primitive_acceleration_structure":
            intersect_args.append(instance_mask)
        if intersection_function_table is not None:
            intersect_args.append(intersection_function_table["name"])
        payload_diagnostic = None
        payload_argument = self.metal_trace_ray_payload_argument(
            raw_args[10], rendered_args[10], intersection_function_table
        )
        if payload_argument["argument"] is not None:
            intersect_args.append(payload_argument["argument"])
        elif payload_argument["reason"] is not None:
            payload_diagnostic = self.unsupported_metal_ray_tracing_intrinsic(
                "TraceRay payload", payload_argument["reason"]
            )
        intersector_type = f"intersector<{intersector_tags}>"
        intersection_type = f"intersection_result<{intersector_tags}>"

        lines = [
            f"ray {ray_name}",
            f"{ray_name}.origin = {origin}",
            f"{ray_name}.direction = {direction}",
            f"{ray_name}.min_distance = {min_distance}",
            f"{ray_name}.max_distance = {max_distance}",
            f"{intersector_type} {intersector_name}",
            (
                f"{intersection_type} {intersection_name} = "
                f"{intersector_name}.intersect({', '.join(intersect_args)})"
            ),
        ]
        if payload_diagnostic is not None:
            lines.append(payload_diagnostic)
        lines.append(f"(void){intersection_name}")
        return "\n".join(lines)

    def metal_trace_ray_payload_argument(
        self, raw_payload_arg, rendered_payload_arg, intersection_function_table
    ):
        if self.is_metal_trace_ray_null_payload(raw_payload_arg):
            return {"argument": None, "reason": None}
        if intersection_function_table is None:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a compatible "
                    "intersection_function_table"
                ),
            }
        payload_display_name = (
            self.assignment_target_display_name(raw_payload_arg)
            or self.expression_name(raw_payload_arg)
            or "<expr>"
        )
        payload_root_name = self.assignment_target_root_name(raw_payload_arg)
        if not payload_root_name or payload_root_name not in self.local_variable_types:
            return {
                "argument": None,
                "reason": "payload forwarding requires a thread-local payload lvalue",
            }
        if payload_root_name in self.current_metal_non_thread_payload_parameters:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local payload lvalue; "
                    "ray payload parameters use a non-thread address space"
                ),
            }
        payload_address_space = self.argument_address_space(raw_payload_arg)
        if payload_address_space not in {None, "thread"}:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local payload lvalue; "
                    f"payload '{payload_display_name}' uses {payload_address_space} "
                    "address space"
                ),
            }
        if payload_root_name in self.current_readonly_metal_parameters:
            readonly_reason = self.current_readonly_metal_parameter_reasons.get(
                payload_root_name, "readonly"
            )
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a mutable thread-local struct "
                    f"payload lvalue; payload '{payload_display_name}' is "
                    f"{readonly_reason}"
                ),
            }
        payload_type = self.expression_result_type(raw_payload_arg)
        if self.metal_type_is_pointer_like(payload_type):
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local struct payload "
                    f"lvalue; payload '{payload_display_name}' has pointer or "
                    "array type"
                ),
            }
        payload_struct_type = self.metal_payload_struct_type_name(payload_type)
        if payload_struct_type is None:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local struct payload "
                    "lvalue"
                ),
            }
        if (
            self.metal_ray_payload_parameter_types
            and payload_struct_type not in self.metal_ray_payload_parameter_types
        ):
            expected_label = " or ".join(sorted(self.metal_ray_payload_parameter_types))
            return {
                "argument": None,
                "reason": (
                    f"ray payload argument '{payload_display_name}' has type "
                    f"'{payload_struct_type}' but declared ray payload interface "
                    f"expects '{expected_label}'"
                ),
            }
        return {"argument": rendered_payload_arg, "reason": None}

    def is_metal_trace_ray_null_payload(self, payload_arg):
        literal_value = self.literal_int_value(payload_arg, self.literal_int_constants)
        return literal_value == 0

    def collect_metal_ray_payload_parameter_types(self, stages):
        payload_types = set()
        if not isinstance(stages, dict):
            return payload_types

        for stage_type, stage in stages.items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in {
                "ray_generation",
                "ray_intersection",
                "ray_any_hit",
                "ray_closest_hit",
                "ray_miss",
                "ray_callable",
                "intersection",
                "anyhit",
                "closesthit",
                "miss",
                "callable",
            }:
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if self.metal_ray_semantic_role(parameter, stage_name) != "payload":
                        continue
                    payload_type = self.metal_parameter_payload_struct_type(parameter)
                    if payload_type is not None:
                        payload_types.add(payload_type)

            for local_var in getattr(stage, "local_variables", []) or []:
                if self.metal_ray_semantic_role(local_var, stage_name) != "payload":
                    continue
                payload_type = self.metal_parameter_payload_struct_type(local_var)
                if payload_type is not None:
                    payload_types.add(payload_type)

        return payload_types

    def metal_ray_semantic_role(self, node, shader_type=None):
        semantic = self.semantic_from_node(node)
        if not semantic:
            return None
        compact = "".join(ch for ch in str(semantic).lower() if ch.isalnum())
        if compact in {
            "payload",
            "raypayload",
            "raypayloadext",
            "raypayloadin",
            "raypayloadinext",
        }:
            return "payload"
        if compact in {"hitattribute", "hitattributeext"}:
            return "hit_attribute"
        if compact in {
            "callabledata",
            "callabledataext",
            "callabledatain",
            "callabledatainext",
        }:
            return "callable_data"
        if compact in {"ray", "rayext", "objectray", "objectrayext"}:
            return "ray"
        return None

    def is_metal_ray_stage(self, shader_type):
        return shader_type in {
            "ray_generation",
            "ray_intersection",
            "ray_any_hit",
            "ray_closest_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        }

    def validate_metal_ray_stage_signature(self, func, shader_type, raw_return_type):
        if not self.is_metal_ray_stage(shader_type):
            return

        self.validate_metal_ray_hit_attribute_parameters(func, shader_type)
        if shader_type == "ray_generation":
            self.validate_metal_ray_generation_parameters(func, shader_type)
        if shader_type in {"ray_intersection", "intersection"}:
            self.validate_metal_ray_intersection_parameters(func, shader_type)
            self.validate_metal_ray_intersection_return_type(func, raw_return_type)

    def validate_metal_ray_generation_parameters(self, func, shader_type):
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        for parameter in parameters:
            parameter_name = getattr(parameter, "name", "<unnamed>")
            if self.metal_ray_semantic_role(parameter, shader_type) == "ray":
                raise ValueError(
                    f"Metal ray_generation parameter '{parameter_name}' cannot "
                    "use @ray; construct Metal ray values inside the kernel"
                )

            raw_type = self.parameter_raw_type(parameter)
            if not self.is_plain_metal_ray_value_parameter_type(raw_type):
                continue
            type_name = self.type_name_string(raw_type) or str(raw_type)
            raise ValueError(
                f"Metal ray_generation parameter '{parameter_name}' has "
                f"unsupported ray value type '{type_name}'; construct ray "
                "values inside the kernel or pass data through an explicitly "
                "bound buffer"
            )

    def validate_metal_ray_intersection_parameters(self, func, shader_type):
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        for parameter in parameters:
            ray_semantic_role = self.metal_ray_semantic_role(parameter, shader_type)
            if ray_semantic_role == "payload":
                continue
            if ray_semantic_role != "ray":
                if self.is_metal_ray_intersection_builtin_parameter(parameter):
                    continue
                if self.is_metal_ray_intersection_buffer_parameter(
                    parameter, shader_type
                ):
                    continue
                parameter_name = getattr(parameter, "name", "<unnamed>")
                raise ValueError(
                    f"Metal ray_intersection parameter '{parameter_name}' is not "
                    "a supported intersection parameter; use @payload, a Metal "
                    "intersection builtin semantic, or an explicitly bound "
                    "device/constant buffer"
                )
                continue
            parameter_name = getattr(parameter, "name", "<unnamed>")
            raise ValueError(
                f"Metal ray_intersection parameter '{parameter_name}' cannot use "
                "@ray; Metal intersection functions do not accept ray input "
                "parameters"
            )

    def is_plain_metal_ray_value_parameter_type(self, raw_type):
        if isinstance(
            raw_type, (PointerType, ReferenceType)
        ) or self.is_array_type_node(raw_type):
            return False

        type_name = self.type_name_string(raw_type)
        if type_name is None:
            return False
        type_name = str(type_name).strip()
        if not type_name:
            return False
        return (
            type_name == "ray"
            or self.requires_metal_builtin_ray_desc(type_name)
            or self.is_metal_ray_query_type_name(type_name)
        )

    def is_metal_ray_intersection_builtin_parameter(self, parameter):
        semantic = self.semantic_from_node(parameter)
        metal_semantic = self.canonical_metal_semantic(semantic)
        if metal_semantic is None:
            return False
        normalized = str(metal_semantic).strip().lower()
        return normalized in {
            "origin",
            "direction",
            "min_distance",
            "max_distance",
            "geometry_id",
            "primitive_id",
            "instance_id",
            "world_space_origin",
            "world_space_direction",
            "barycentric_coord",
            "object_to_world_transform",
            "world_to_object_transform",
        }

    def is_metal_ray_intersection_buffer_parameter(self, parameter, shader_type):
        raw_type = self.parameter_raw_type(parameter)
        if not (
            isinstance(raw_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_type)
        ):
            return False
        if self.explicit_buffer_binding_index(parameter) is None:
            return False
        address_space = self.normalized_address_space(
            self.effective_parameter_address_space(
                raw_type,
                parameter,
                shader_type,
                default_for_stage_binding=False,
            )
        )
        return address_space in {"device", "constant"}

    def validate_metal_kernel_return_type(self, func, shader_type, raw_return_type):
        if self.map_type(raw_return_type) == "void":
            return
        function_name = getattr(func, "name", "<entry>")
        raise ValueError(
            f"Metal {shader_type} entry point '{function_name}' must return void; "
            "kernel functions cannot return values"
        )

    def validate_metal_ray_hit_attribute_parameters(self, func, shader_type):
        allowed_stages = {"ray_any_hit", "ray_closest_hit", "anyhit", "closesthit"}
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        for parameter in parameters:
            if not self.is_metal_hit_attribute_parameter(parameter):
                continue
            if shader_type in allowed_stages:
                continue
            parameter_name = getattr(parameter, "name", "<unnamed>")
            raise ValueError(
                "Metal @hit_attribute parameter "
                f"'{parameter_name}' is only valid on any-hit or closest-hit stages"
            )

    def validate_metal_ray_intersection_return_type(self, func, raw_return_type):
        primitive_type = self.metal_ray_intersection_primitive_type(
            func, raw_return_type
        )
        mapped_return_type = self.map_type(raw_return_type)
        if primitive_type == "triangle" and mapped_return_type == "bool":
            return

        members = self.metal_intersection_result_members(raw_return_type)
        if members is None:
            if primitive_type == "triangle":
                raise ValueError(
                    "Metal triangle intersection stage must return bool or a "
                    "result struct"
                )
            raise ValueError(
                "Metal "
                f"{primitive_type} intersection stage must return a result struct"
            )

        self.require_metal_intersection_result_member(
            members,
            "accept_intersection",
            "bool",
            primitive_type,
            raw_return_type,
        )
        if primitive_type == "bounding_box":
            self.require_metal_intersection_result_member(
                members,
                "distance",
                "float",
                primitive_type,
                raw_return_type,
            )

    def require_metal_intersection_result_member(
        self, members, role, expected_type, primitive_type, raw_return_type
    ):
        for member in members:
            if self.metal_intersection_result_member_role(member) != role:
                continue
            member_type = self.type_name_string(
                getattr(
                    member,
                    "member_type",
                    getattr(member, "vtype", getattr(member, "var_type", None)),
                )
            )
            if self.map_type(member_type) == expected_type:
                return

        raise ValueError(
            "Metal "
            f"{primitive_type} intersection result '{raw_return_type}' requires "
            f"a {expected_type} @{role} member"
        )

    def metal_intersection_result_members(self, raw_return_type):
        struct_type = self.metal_payload_struct_type_name(raw_return_type)
        if struct_type is None:
            return None
        struct_node = self.structs_by_name.get(struct_type)
        if struct_node is None:
            return None
        return getattr(struct_node, "members", []) or []

    def metal_intersection_result_member_role(self, member):
        semantic = self.semantic_from_node(member)
        if not semantic:
            return None
        compact = "".join(ch for ch in str(semantic).lower() if ch.isalnum())
        if compact in {"acceptintersection", "acceptintersectionext"}:
            return "accept_intersection"
        if compact in {"distance", "hitdistance", "intersectiondistance"}:
            return "distance"
        return None

    def is_metal_hit_attribute_parameter(self, parameter):
        return self.metal_ray_semantic_role(parameter) == "hit_attribute"

    def metal_parameter_payload_struct_type(self, parameter):
        type_name = self.type_name_string(
            getattr(
                parameter,
                "param_type",
                getattr(parameter, "var_type", getattr(parameter, "vtype", None)),
            )
        )
        return self.metal_payload_struct_type_name(type_name)

    def metal_acceleration_structure_argument_type(self, expr):
        name = self.expression_name(expr)
        local_type = self.local_variable_types.get(name)
        if local_type and self.is_acceleration_structure_type(local_type):
            return self.map_resource_type_with_format(local_type)
        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            _,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) == name:
                return mapped_type
        return "instance_acceleration_structure"

    def metal_trace_ray_acceleration_structure_argument(self, raw_arg, rendered_arg):
        display_name = (
            self.assignment_target_display_name(raw_arg)
            or self.expression_name(raw_arg)
            or rendered_arg
            or "<expr>"
        )

        unsupported_array = self.unsupported_metal_acceleration_structure_array_reason(
            raw_arg
        )
        if unsupported_array is not None:
            return {
                "argument": None,
                "type": None,
                "reason": (
                    "acceleration structure argument "
                    f"'{display_name}' uses an acceleration_structure array; "
                    f"{unsupported_array}; TraceRay requires a single "
                    "acceleration_structure resource"
                ),
            }

        result_type = self.expression_result_type(raw_arg)
        if result_type is not None:
            return self.metal_trace_ray_acceleration_structure_argument_from_type(
                raw_arg, rendered_arg, display_name, result_type
            )

        name = self.expression_name(raw_arg)
        local_type = self.local_variable_types.get(name)
        if local_type is not None:
            return self.metal_trace_ray_acceleration_structure_argument_from_type(
                raw_arg, rendered_arg, display_name, local_type
            )

        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            array_size,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) != name:
                continue
            if array_size is not None and not isinstance(raw_arg, ArrayAccessNode):
                return {
                    "argument": None,
                    "type": None,
                    "reason": (
                        "acceleration structure argument "
                        f"'{display_name}' has pointer or array type; "
                        "TraceRay requires an acceleration_structure element"
                    ),
                }
            return {"argument": rendered_arg, "type": mapped_type, "reason": None}

        return {
            "argument": None,
            "type": None,
            "reason": (
                "acceleration structure argument "
                f"'{display_name}' must resolve to an acceleration_structure "
                "resource"
            ),
        }

    def metal_trace_ray_acceleration_structure_argument_from_type(
        self, raw_arg, rendered_arg, display_name, raw_type
    ):
        type_name = self.type_name_string(raw_type) or "<unknown>"
        if not self.is_acceleration_structure_type(raw_type):
            return {
                "argument": None,
                "type": None,
                "reason": (
                    "acceleration structure argument "
                    f"'{display_name}' has type '{type_name}' but TraceRay "
                    "requires an acceleration_structure resource"
                ),
            }

        if self.metal_type_is_pointer_like(raw_type):
            return {
                "argument": None,
                "type": None,
                "reason": (
                    "acceleration structure argument "
                    f"'{display_name}' has pointer or array type; TraceRay "
                    "requires an acceleration_structure element"
                ),
            }

        return {
            "argument": rendered_arg,
            "type": self.map_resource_type_with_format(raw_type),
            "reason": None,
        }

    def default_metal_intersector_tags(self, acceleration_structure_type):
        if acceleration_structure_type == "primitive_acceleration_structure":
            return "triangle_data"
        return "triangle_data, instancing"

    def default_intersection_function_table(self, acceleration_structure_type):
        candidates = []
        for (
            intersection_function_table_variable,
            _,
            intersection_function_table_type,
            _,
        ) in self.intersection_function_table_variables:
            name = getattr(intersection_function_table_variable, "name", None)
            tags = self.intersection_function_table_tags(
                intersection_function_table_type
            )
            if (
                name
                and tags
                and self.intersection_function_table_tags_match_acceleration_structure(
                    tags, acceleration_structure_type
                )
            ):
                candidates.append({"name": name, "tags": ", ".join(tags)})
        if len(candidates) == 1:
            return candidates[0]
        return None

    def intersection_function_table_tags(self, table_type):
        type_name = str(table_type)
        if "<" not in type_name or not type_name.endswith(">"):
            return []
        tags = type_name.split("<", 1)[1][:-1].strip()
        if not tags:
            return []
        return [tag.strip() for tag in tags.split(",") if tag.strip()]

    def intersection_function_table_tags_match_acceleration_structure(
        self, tags, acceleration_structure_type
    ):
        has_instancing = "instancing" in tags
        if acceleration_structure_type == "primitive_acceleration_structure":
            return not has_instancing
        return has_instancing

    def generate_metal_call_shader(self, raw_args, rendered_args):
        if len(raw_args) == 2:
            table = self.default_visible_function_table()
            if table is None:
                return None
            table_name = table["name"]
            table_type = table["type"]
            shader_index, callable_data = rendered_args
            raw_callable_data = raw_args[1]
        elif len(raw_args) == 3:
            unsupported_array = self.unsupported_metal_ray_function_table_array_reason(
                raw_args[0]
            )
            if unsupported_array is not None:
                return self.unsupported_metal_ray_tracing_intrinsic(
                    "CallShader", unsupported_array
                )
            table_name = rendered_args[0]
            table_type = self.visible_function_table_argument_type(raw_args[0])
            if not self.is_visible_function_table_type(table_type):
                table_display_name = (
                    self.assignment_target_display_name(raw_args[0])
                    or self.expression_name(raw_args[0])
                    or table_name
                    or "<expr>"
                )
                return self.unsupported_metal_ray_tracing_intrinsic(
                    "CallShader",
                    "explicit table argument "
                    f"'{table_display_name}' must be a "
                    "visible_function_table resource",
                )
            shader_index = rendered_args[1]
            callable_data = rendered_args[2]
            raw_callable_data = raw_args[2]
        else:
            return None

        callable_data_argument = self.metal_call_shader_callable_data_argument(
            raw_callable_data, callable_data, table_type
        )
        if callable_data_argument["reason"] is not None:
            return self.unsupported_metal_ray_tracing_intrinsic(
                "CallShader callable data", callable_data_argument["reason"]
            )

        callable_data = callable_data_argument["argument"]
        return f"{table_name}[{shader_index}]({callable_data})"

    def default_visible_function_table(self):
        tables = [
            {"name": getattr(variable, "name", None), "type": table_type}
            for variable, _, table_type, _ in self.visible_function_table_variables
            if getattr(variable, "name", None)
        ]
        if len(tables) == 1:
            return tables[0]
        return None

    def default_visible_function_table_name(self):
        table = self.default_visible_function_table()
        if table is None:
            return None
        return table["name"]

    def visible_function_table_argument_type(self, table_arg):
        table_name = self.expression_name(table_arg)
        for variable, _, table_type, _ in self.visible_function_table_variables:
            if getattr(variable, "name", None) == table_name:
                return table_type
        if table_name in self.local_variable_types:
            return self.local_variable_types[table_name]
        return self.expression_result_type(table_arg)

    def metal_call_shader_callable_data_argument(
        self, raw_callable_data, rendered_callable_data, table_type
    ):
        callable_data_name = (
            self.assignment_target_display_name(raw_callable_data) or "<expr>"
        )
        root_name = self.assignment_target_root_name(raw_callable_data)
        if not root_name or root_name not in self.local_variable_types:
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local "
                    "callable-data lvalue"
                ),
            }

        callable_data_address_space = self.argument_address_space(raw_callable_data)
        if callable_data_address_space not in {None, "thread"}:
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local "
                    "callable-data lvalue; callable data "
                    f"'{callable_data_name}' uses {callable_data_address_space} "
                    "address space"
                ),
            }

        if root_name in self.current_readonly_metal_parameters:
            readonly_reason = self.current_readonly_metal_parameter_reasons.get(
                root_name, "readonly"
            )
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a mutable thread-local "
                    "callable-data lvalue; callable data "
                    f"'{callable_data_name}' is {readonly_reason}"
                ),
            }

        callable_data_raw_type = self.expression_result_type(raw_callable_data)
        if self.metal_type_is_pointer_like(callable_data_raw_type):
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local struct "
                    f"callable-data lvalue; callable data '{callable_data_name}' "
                    "has pointer or array type"
                ),
            }

        callable_data_type = self.metal_payload_struct_type_name(callable_data_raw_type)
        if callable_data_type is None:
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local struct "
                    "callable-data lvalue"
                ),
            }

        table_payload_type = self.visible_function_table_payload_type(table_type)
        if table_payload_type is not None and table_payload_type != callable_data_type:
            return {
                "argument": None,
                "reason": (
                    f"visible_function_table expects callable data "
                    f"'{table_payload_type}' but argument '{callable_data_name}' "
                    f"has '{callable_data_type}'"
                ),
            }

        return {"argument": rendered_callable_data, "reason": None}

    def visible_function_table_payload_type(self, table_type):
        type_name = self.type_name_string(table_type)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return None
        table_kind, payload_type = type_name.split("<", 1)
        if table_kind not in {"visible_function_table", "visibleFunctionTable"}:
            return None
        payload_type = payload_type[:-1].strip()
        if payload_type.startswith("void(") and payload_type.endswith(")"):
            payload_type = payload_type[5:-1].strip()
            if not payload_type:
                return None
            payload_type = payload_type.split(",", 1)[0].strip()
        return self.metal_payload_struct_type_name(payload_type)

    def metal_payload_struct_type_name(self, payload_type):
        type_name = self.type_name_string(payload_type)
        if not type_name:
            return None
        type_name = self.strip_metal_type_qualifiers(type_name)
        if "[" in type_name and "]" in type_name:
            type_name, _ = split_array_type_suffix(type_name)
        return type_name if type_name in self.structs_by_name else None

    def strip_metal_type_qualifiers(self, type_name):
        type_name = str(type_name).strip()
        qualifiers = {
            "const",
            "constant",
            "device",
            "object_data",
            "ray_data",
            "thread",
            "threadgroup",
        }
        changed = True
        while changed:
            changed = False
            parts = type_name.split(None, 1)
            if len(parts) == 2 and parts[0] in qualifiers:
                type_name = parts[1].strip()
                changed = True
        while type_name.endswith("&") or type_name.endswith("*"):
            type_name = type_name[:-1].strip()
        return type_name

    def unsupported_metal_ray_tracing_intrinsic(self, operation, reason):
        return f"/* unsupported Metal ray tracing intrinsic: {operation} - {reason} */"

    def unsupported_metal_acceleration_structure_array_diagnostic(self, name):
        return (
            "/* unsupported Metal ray tracing resource: arrays of "
            "acceleration_structure are not valid Metal buffer parameters "
            f"({name}) */\n"
        )

    def unsupported_metal_acceleration_structure_array_alias_diagnostic(
        self, name, source_name
    ):
        return (
            "/* unsupported Metal ray tracing resource: local "
            f"acceleration_structure alias '{name}' cannot be initialized from "
            f"unsupported acceleration_structure array '{source_name}' */\n"
        )

    def unsupported_metal_acceleration_structure_array_alias_source(self, expr):
        source_name = self.expression_name(expr)
        if source_name in self.unsupported_metal_acceleration_structure_array_variables:
            return source_name
        return None

    def unsupported_metal_acceleration_structure_array_reason(self, expr):
        resource_name = self.expression_name(expr)
        if (
            resource_name
            not in self.unsupported_metal_acceleration_structure_array_variables
        ):
            return None
        return (
            "arrays of acceleration_structure are not valid Metal buffer "
            f"parameters ({resource_name})"
        )

    def unsupported_metal_acceleration_structure_array_expression(self, expr):
        reason = self.unsupported_metal_acceleration_structure_array_reason(expr)
        if reason is None:
            return None
        return f"0 /* unsupported Metal ray tracing resource: {reason} */"

    def metal_acceleration_structure_array_parameter_kind(self, vtype, node=None):
        if not self.is_acceleration_structure_type(vtype):
            return None
        array_resource = self.resource_array_parameter(vtype, node)
        if array_resource is None:
            return None
        resource_type, _array_size = array_resource
        return resource_type

    def unsupported_metal_ray_function_table_array_diagnostic(self, table_kind, name):
        return (
            f"/* unsupported Metal ray tracing resource: arrays of {table_kind} "
            f"are not valid Metal buffer parameters ({name}) */\n"
        )

    def unsupported_metal_ray_function_table_array_alias_diagnostic(
        self, name, source_name
    ):
        table_kind = self.unsupported_metal_ray_function_table_array_variables.get(
            source_name, "ray function table"
        )
        return (
            "/* unsupported Metal ray tracing resource: local "
            f"{table_kind} alias '{name}' cannot be initialized from "
            f"unsupported {table_kind} array '{source_name}' */\n"
        )

    def metal_ray_function_table_kind(self, vtype):
        if self.is_visible_function_table_type(vtype):
            return "visible_function_table"
        if self.is_intersection_function_table_type(vtype):
            return "intersection_function_table"
        return None

    def metal_ray_function_table_array_parameter_kind(self, vtype, node=None):
        table_kind = self.metal_ray_function_table_kind(vtype)
        if table_kind is None:
            return None
        if self.resource_array_parameter(vtype, node) is None:
            return None
        return table_kind

    def unsupported_metal_ray_function_table_array_alias_source(self, expr):
        source_name = self.expression_name(expr)
        if source_name in self.unsupported_metal_ray_function_table_array_variables:
            return source_name
        return None

    def unsupported_metal_ray_function_table_array_reason(self, expr):
        table_name = self.expression_name(expr)
        table_kind = self.unsupported_metal_ray_function_table_array_variables.get(
            table_name
        )
        if table_kind is None:
            return None
        return (
            f"arrays of {table_kind} are not valid Metal buffer parameters "
            f"({table_name})"
        )

    def unsupported_metal_ray_function_table_array_expression(self, expr):
        reason = self.unsupported_metal_ray_function_table_array_reason(expr)
        if reason is None:
            return None
        return f"0 /* unsupported Metal ray tracing resource: {reason} */"

    def unsupported_metal_ray_function_table_array_member_call(self, func_expr):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        object_expr = getattr(
            func_expr, "object_expr", getattr(func_expr, "object", None)
        )
        return self.unsupported_metal_ray_function_table_array_expression(object_expr)

    def generate_atomic_function_call(self, func_name, args):
        if func_name in self.user_function_names:
            return None
        if not self.is_metal_atomic_function_name(func_name) or not args:
            return None

        args = list(args)
        rendered_args = [self.generate_expression(arg) for arg in args]
        args, rendered_args = self.strip_metal_atomic_memory_scope_argument(
            func_name, args, rendered_args
        )
        if self.metal_atomic_target_needs_address(args[0], rendered_args[0]):
            rendered_args[0] = f"&{rendered_args[0]}"
        if self.is_metal_atomic_compare_exchange_name(func_name) and len(args) >= 2:
            expected_diagnostic = (
                self.metal_compare_exchange_expected_address_space_diagnostic(args[1])
            )
            if expected_diagnostic is not None:
                return expected_diagnostic
            if self.metal_compare_exchange_expected_needs_address(
                args[1], rendered_args[1]
            ):
                rendered_args[1] = f"&{rendered_args[1]}"
        return (
            f"{self.metal_atomic_intrinsic_name(func_name)}({', '.join(rendered_args)})"
        )

    def generate_metal_buffer_resource_atomic_call(self, func_name, args):
        if func_name in self.user_function_names:
            return None
        operation_info = self.buffer_atomic_operations().get(func_name)
        if operation_info is None:
            return None

        operation, expected_arity = operation_info
        if len(args) != expected_arity:
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name,
                args[0] if args else None,
                f"requires {expected_arity} argument(s), got {len(args)}",
            )
        if operation == "compare_exchange":
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name,
                args[0],
                "compare-exchange requires explicit Metal atomic storage",
            )

        target = args[0]
        target_type = self.expression_result_type(target)
        mapped_target_type = self.map_type(target_type)
        if mapped_target_type not in {"int", "uint"}:
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name,
                target,
                (
                    "requires a scalar int or uint device/threadgroup target, "
                    f"got {mapped_target_type or 'unknown'}"
                ),
            )

        address_space = self.argument_address_space(target)
        if address_space not in {"device", "threadgroup"}:
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name,
                target,
                "requires a device or threadgroup target",
            )

        target_expr = self.generate_expression(target)
        if not self.is_metal_address_expression(target, target_expr):
            target_expr = f"&{target_expr}"
        value = self.generate_expression_with_expected(args[1], mapped_target_type)
        atomic_type = f"atomic_{mapped_target_type}"
        atomic_target = (
            f"reinterpret_cast<{address_space} {atomic_type}*>({target_expr})"
        )
        return (
            f"atomic_{operation}_explicit("
            f"{atomic_target}, {value}, memory_order_relaxed)"
        )

    def unsupported_metal_buffer_resource_atomic_call(self, func_name, target, reason):
        return_type = (
            self.expression_result_type(target) or self.current_expression_expected_type
        )
        zero_value = self.diagnostic_zero_value_for_type(return_type or "int")
        return (
            f"/* unsupported Metal buffer atomic: {func_name} {reason} */ {zero_value}"
        )

    def strip_metal_atomic_memory_scope_argument(self, func_name, args, rendered_args):
        expected_count = self.metal_atomic_expected_argument_count(func_name)
        if (
            expected_count is not None
            and len(args) == expected_count + 1
            and self.is_metal_atomic_memory_scope_argument(args[-1], rendered_args[-1])
        ):
            return args[:-1], rendered_args[:-1]
        return args, rendered_args

    def metal_atomic_expected_argument_count(self, func_name):
        if self.is_metal_atomic_compare_exchange_name(func_name):
            return 5
        if func_name == "atomic_load_explicit":
            return 2
        if func_name in {
            "atomic_fetch_add_explicit",
            "atomic_fetch_sub_explicit",
            "atomic_fetch_min_explicit",
            "atomic_fetch_max_explicit",
            "atomic_fetch_and_explicit",
            "atomic_fetch_or_explicit",
            "atomic_fetch_xor_explicit",
            "atomic_store_explicit",
            "atomic_exchange_explicit",
        }:
            return 3
        return None

    def is_metal_atomic_memory_scope_argument(self, arg, rendered_arg):
        scope_names = {
            "memory_scope_device",
            "memory_scope_grid",
            "memory_scope_invocation",
            "memory_scope_queuefamily",
            "memory_scope_queue_family",
            "memory_scope_simdgroup",
            "memory_scope_subgroup",
            "memory_scope_system",
            "memory_scope_thread",
            "memory_scope_threadgroup",
            "memory_scope_workgroup",
        }
        for candidate in (self.expression_name(arg), rendered_arg):
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            normalized = normalized.replace(" ", "")
            normalized = normalized.replace("::", "_").replace(".", "_")
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized in scope_names:
                return True
        return False

    def metal_atomic_target_needs_address(self, arg, rendered_arg):
        if self.is_metal_address_expression(arg, rendered_arg):
            return False
        target_type = self.expression_result_type(arg)
        if self.metal_type_is_pointer_like(target_type):
            return False
        return self.is_metal_atomic_value_type(target_type)

    def metal_compare_exchange_expected_needs_address(self, arg, rendered_arg):
        if self.is_metal_address_expression(arg, rendered_arg):
            return False
        expected_type = self.expression_result_type(arg)
        if self.metal_type_is_pointer_like(expected_type):
            return False
        if isinstance(arg, (ArrayAccessNode, MemberAccessNode)):
            return True
        return self.expression_name(arg) is not None

    def metal_compare_exchange_expected_address_space_diagnostic(self, arg):
        member_address_space = self.address_space_qualified_member_address_space(arg)
        if member_address_space is not None:
            address_space = member_address_space
            storage_name = self.assignment_target_display_name(arg)
        else:
            storage_name = self.atomic_expected_storage_root_name(arg)
            if storage_name is None:
                return None
            address_space = self.current_address_space_variables.get(storage_name)
            if address_space is None and storage_name in self.local_variable_types:
                address_space = "thread"
        if address_space in {None, "thread"}:
            return None
        return (
            "false /* unsupported Metal atomic compare-exchange expected pointer: "
            f"expected storage '{storage_name}' uses {address_space} address space; "
            "Metal requires thread storage */"
        )

    def atomic_expected_storage_root_name(self, arg):
        if isinstance(arg, UnaryOpNode) and getattr(arg, "operator", None) == "&":
            return self.atomic_expected_storage_root_name(arg.operand)
        if isinstance(arg, BinaryOpNode) and getattr(arg, "operator", None) in {
            "+",
            "-",
        }:
            return self.atomic_expected_storage_root_name(
                getattr(arg, "left", None)
            ) or self.atomic_expected_storage_root_name(getattr(arg, "right", None))
        return self.assignment_target_root_name(arg)

    def is_metal_address_expression(self, arg, rendered_arg):
        if isinstance(arg, UnaryOpNode) and getattr(arg, "operator", None) == "&":
            return True
        return str(rendered_arg).lstrip().startswith("&")

    def metal_type_is_pointer_like(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return False
        type_name = str(type_name).strip()
        return type_name.endswith("*") or "[" in type_name

    def is_metal_atomic_compare_exchange_name(self, func_name):
        return func_name in {
            "atomic_compare_exchange_weak_explicit",
            "atomic_compare_exchange_strong_explicit",
        }

    def metal_atomic_intrinsic_name(self, func_name):
        if func_name == "atomic_compare_exchange_strong_explicit":
            return "atomic_compare_exchange_weak_explicit"
        return func_name

    def is_metal_atomic_function_name(self, func_name):
        return func_name in {
            "atomic_fetch_add_explicit",
            "atomic_fetch_sub_explicit",
            "atomic_fetch_min_explicit",
            "atomic_fetch_max_explicit",
            "atomic_fetch_and_explicit",
            "atomic_fetch_or_explicit",
            "atomic_fetch_xor_explicit",
            "atomic_load_explicit",
            "atomic_store_explicit",
            "atomic_exchange_explicit",
            "atomic_compare_exchange_weak_explicit",
            "atomic_compare_exchange_strong_explicit",
        }

    def generate_metal_mesh_output_call(self, func_name, args):
        mesh_output = self.current_metal_mesh_output_config
        output_parameter = self.current_metal_mesh_output_parameter
        if not mesh_output or not output_parameter:
            return None

        if func_name == "SetVertex" and len(args) == 2:
            index = self.generate_expression(args[0])
            value = self.metal_mesh_vertex_output_value(args[1], mesh_output)
            return f"{output_parameter}.set_vertex({index}, {value})"

        if func_name == "SetPrimitive" and len(args) == 2:
            primitive_output = self.metal_mesh_primitive_output_call(
                output_parameter, args[0], args[1], mesh_output
            )
            if primitive_output is not None:
                return primitive_output
            return self.metal_mesh_index_output_write(
                output_parameter, func_name, args[0], args[1]
            )

        if func_name == "SetIndex" and len(args) == 2:
            return self.metal_mesh_index_output_write(
                output_parameter, func_name, args[0], args[1]
            )

        return None

    def generate_metal_mesh_output_assignment(
        self, target, rendered_value, operator, value_expr
    ):
        target_info = self.metal_mesh_output_assignment_target(target)
        if target_info is None:
            return None

        compound_member_assignment = (
            operator != "="
            and target_info["member"] is not None
            and target_info["role"] in {"vertices", "primitives"}
        )
        if operator != "=" and not compound_member_assignment:
            return self.unsupported_metal_mesh_output_assignment_diagnostic(
                target_info, "compound mesh output assignments are not supported"
            )

        role = target_info["role"]
        index_expr = target_info["index"]
        if role == "vertices":
            if target_info["member"] is not None:
                return self.generate_metal_mesh_single_member_output_assignment(
                    target_info, rendered_value, "set_vertex", operator
                )
            value = self.metal_mesh_vertex_output_value_from_rendered(
                value_expr, rendered_value, self.current_metal_mesh_output_config
            )
            index = self.generate_expression(index_expr)
            return f"{self.current_metal_mesh_output_parameter}.set_vertex({index}, {value})"

        if role == "indices":
            if target_info["member"] is not None:
                return self.unsupported_metal_mesh_output_assignment_diagnostic(
                    target_info, "index vector components must be written as a whole"
                )
            return self.metal_mesh_index_output_write(
                self.current_metal_mesh_output_parameter,
                "SetPrimitive",
                index_expr,
                value_expr,
            )

        if role == "primitives":
            if target_info["member"] is not None:
                return self.generate_metal_mesh_single_member_output_assignment(
                    target_info, rendered_value, "set_primitive", operator
                )
            index = self.generate_expression(index_expr)
            return (
                f"{self.current_metal_mesh_output_parameter}"
                f".set_primitive({index}, {rendered_value})"
            )

        return None

    def generate_metal_mesh_single_member_output_assignment(
        self, target_info, rendered_value, setter, operator="="
    ):
        output_info = target_info["output"]
        member_name = target_info["member"]
        element_type = output_info["element_type"]
        member_types = self.struct_member_types.get(element_type, {})
        if operator != "=" or list(member_types) != [member_name]:
            accumulator = self.metal_mesh_output_accumulator(target_info)
            if accumulator is None:
                reason = (
                    "compound member writes require an output accumulator"
                    if operator != "="
                    else "partial member writes require an output accumulator"
                )
                return self.unsupported_metal_mesh_output_assignment_diagnostic(
                    target_info, reason
                )
            index = self.generate_expression(target_info["index"])
            temp_name = accumulator["name"]
            return (
                f"{temp_name}.{member_name} {operator} {rendered_value}\n"
                f"{self.current_metal_mesh_output_parameter}"
                f".{setter}({index}, {temp_name})"
            )

        index = self.generate_expression(target_info["index"])
        value = f"{element_type}{{{rendered_value}}}"
        return (
            f"{self.current_metal_mesh_output_parameter}" f".{setter}({index}, {value})"
        )

    def collect_metal_mesh_output_accumulators(self, func, reserved_parameter_names):
        if not self.current_metal_mesh_output_config:
            return {}

        accumulators = {}
        reserved_names = set(reserved_parameter_names or ())
        reserved_names.add(self.current_metal_mesh_output_parameter)
        reserved_names.update(self.metal_function_local_variable_names(func))

        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, AssignmentNode):
                continue
            target = getattr(node, "target", getattr(node, "left", None))
            target_info = self.metal_mesh_output_assignment_target(target)
            if target_info is None or target_info["member"] is None:
                continue
            if target_info["role"] not in {"vertices", "primitives"}:
                continue

            element_type = target_info["output"]["element_type"]
            member_types = self.struct_member_types.get(element_type, {})
            operator = getattr(node, "operator", "=")
            if operator == "=" and list(member_types) == [target_info["member"]]:
                continue

            key = self.metal_mesh_output_accumulator_key(target_info)
            if key in accumulators:
                continue

            temp_base = (
                f"_crossglMesh{target_info['role'].title()}"
                f"_{target_info['name']}"
                f"_{self.metal_identifier_suffix(key[2])}"
            )
            temp_name = self.unique_metal_generated_name(temp_base, reserved_names)
            reserved_names.add(temp_name)
            accumulators[key] = {
                "name": temp_name,
                "type": element_type,
            }
        return accumulators

    def generate_metal_mesh_output_accumulator_declarations(self):
        lines = []
        for accumulator in self.current_metal_mesh_output_accumulators.values():
            lines.append(f"    {accumulator['type']} {accumulator['name']} = {{}};\n")
        return "".join(lines)

    def metal_mesh_output_accumulator(self, target_info):
        return self.current_metal_mesh_output_accumulators.get(
            self.metal_mesh_output_accumulator_key(target_info)
        )

    def metal_mesh_output_accumulator_key(self, target_info):
        index_expr = target_info.get("index")
        index_text = self.safe_expression_to_string(index_expr) if index_expr else "0"
        return (target_info["role"], target_info["name"], index_text)

    def metal_identifier_suffix(self, value):
        suffix = "".join(
            ch if (ch.isalnum() or ch == "_") else "_" for ch in str(value)
        ).strip("_")
        if not suffix:
            suffix = "value"
        if suffix[0].isdigit():
            suffix = f"i_{suffix}"
        return suffix

    def metal_mesh_output_assignment_target(self, target):
        mesh_output = self.current_metal_mesh_output_config
        if not mesh_output:
            return None
        outputs = mesh_output.get("output_parameters", {})

        member = None
        access = target
        if isinstance(target, MemberAccessNode):
            member = str(target.member)
            access = getattr(target, "object", None)

        if not isinstance(access, ArrayAccessNode):
            return None
        array_expr = getattr(access, "array", None)
        parameter_name = self.expression_name(array_expr)
        output = outputs.get(parameter_name)
        if output is None:
            return None

        return {
            "name": parameter_name,
            "role": output["role"],
            "output": output,
            "index": getattr(access, "index", None),
            "member": member,
        }

    def unsupported_metal_mesh_output_assignment_diagnostic(self, target_info, reason):
        name = target_info.get("name", "<unknown>")
        role = target_info.get("role", "output")
        return (
            "/* unsupported Metal mesh output assignment: "
            f"{role} output '{name}' - {reason} */"
        )

    def metal_mesh_vertex_output_value(self, expr, mesh_output):
        value = self.generate_expression(expr)
        return self.metal_mesh_vertex_output_value_from_rendered(
            expr, value, mesh_output
        )

    def metal_mesh_vertex_output_value_from_rendered(self, expr, value, mesh_output):
        value_type = self.expression_result_type(expr)
        mapped_type = self.map_type(value_type) if value_type else None
        vertex_type = mesh_output["vertex_type"]

        if mapped_type == vertex_type:
            return value
        if mapped_type == "float3":
            return f"{vertex_type}{{float4({value}, 1.0)}}"
        return f"{vertex_type}{{{value}}}"

    def metal_mesh_primitive_output_call(
        self, output_parameter, index_expr, value_expr, mesh_output
    ):
        primitive_type = mesh_output.get("primitive_type")
        if not primitive_type or primitive_type == "void":
            return None

        value_type = self.expression_result_type(value_expr)
        mapped_type = self.map_type(value_type) if value_type else None
        if mapped_type != primitive_type:
            return None

        index = self.generate_expression(index_expr)
        value = self.generate_expression(value_expr)
        return f"{output_parameter}.set_primitive({index}, {value})"

    def metal_mesh_index_output_write(
        self, output_parameter, func_name, index_expr, value_expr
    ):
        index = self.generate_expression(index_expr)
        value = self.generate_expression(value_expr)
        vector_width = self.metal_mesh_index_vector_width(value_expr)
        if vector_width is None:
            return f"{output_parameter}.set_index({index}, {value})"

        base_index = index
        if func_name == "SetPrimitive":
            base_index = f"({index}) * {vector_width}"

        if vector_width in {2, 4}:
            return (
                f"{output_parameter}.set_indices("
                f"{base_index}, uchar{vector_width}({value}))"
            )

        if vector_width == 3:
            return "\n".join(
                f"{output_parameter}.set_index("
                f"{self.metal_mesh_index_component_expression(base_index, offset)}, "
                f"{value}.{component})"
                for offset, component in enumerate(("x", "y", "z"))
            )

        return f"{output_parameter}.set_index({index}, {value})"

    def metal_mesh_index_vector_width(self, expr):
        value_type = self.expression_result_type(expr)
        if not value_type:
            return None
        mapped_type = self.map_type(value_type)
        if (
            len(mapped_type) < 2
            or mapped_type[-1] not in {"2", "3", "4"}
            or mapped_type[:-1] not in {"bool", "int", "uint"}
        ):
            return None
        return int(mapped_type[-1])

    def metal_mesh_index_component_expression(self, base_index, offset):
        if offset == 0:
            return base_index
        return f"({base_index}) + {offset}"

    def generate_mesh_op_expression(self, expr):
        arguments = self.normalized_metal_intrinsic_args(expr.arguments)
        if (
            expr.operation == "SetMeshOutputCounts"
            and self.current_metal_mesh_output_parameter
            and len(arguments) >= 2
        ):
            primitive_count = self.generate_expression(arguments[1])
            return (
                f"{self.current_metal_mesh_output_parameter}"
                f".set_primitive_count({primitive_count})"
            )
        if expr.operation == "DispatchMesh":
            if (
                len(arguments) == 4
                and self.current_metal_mesh_payload_parameter is None
            ):
                return (
                    "/* unsupported Metal mesh dispatch: DispatchMesh payload "
                    "argument requires an object_data payload context */"
                )
            if self.current_metal_mesh_grid_properties_parameter is None:
                return (
                    "/* unsupported Metal mesh dispatch: DispatchMesh requires "
                    "mesh_grid_properties context */"
                )
        if (
            expr.operation == "DispatchMesh"
            and self.current_metal_mesh_grid_properties_parameter
            and len(arguments) == 4
        ):
            grid_assignment = self.metal_dispatch_mesh_grid_assignment(arguments[:3])
            if grid_assignment is None:
                return self.unsupported_metal_mesh_dispatch(
                    self.metal_dispatch_mesh_grid_argument_reason(arguments[:3])
                )
            payload_assignment = self.metal_dispatch_mesh_payload_assignment(
                arguments[3]
            )
            return "\n".join([payload_assignment, grid_assignment])
        if (
            expr.operation == "DispatchMesh"
            and self.current_metal_mesh_grid_properties_parameter
            and len(arguments) == 3
        ):
            grid_assignment = self.metal_dispatch_mesh_grid_assignment(arguments)
            if grid_assignment is None:
                return self.unsupported_metal_mesh_dispatch(
                    self.metal_dispatch_mesh_grid_argument_reason(arguments)
                )
            return grid_assignment
        if (
            expr.operation == "DispatchMesh"
            and self.current_metal_mesh_grid_properties_parameter
            and len(arguments) == 1
        ):
            grid_assignment = self.metal_dispatch_mesh_grid_assignment(arguments)
            if grid_assignment is None:
                return self.unsupported_metal_mesh_dispatch(
                    self.metal_dispatch_mesh_grid_argument_reason(arguments)
                )
            return grid_assignment
        return None

    def metal_dispatch_mesh_grid_assignment(self, grid_args):
        reason = self.metal_dispatch_mesh_grid_argument_reason(grid_args)
        if reason is not None:
            return None
        if len(grid_args) == 1:
            grid = self.generate_expression(grid_args[0])
        else:
            grid = "uint3({})".format(
                ", ".join(self.generate_expression(argument) for argument in grid_args)
            )
        return (
            f"{self.current_metal_mesh_grid_properties_parameter}"
            f".set_threadgroups_per_grid({grid})"
        )

    def metal_dispatch_mesh_grid_argument_reason(self, grid_args):
        if len(grid_args) == 1:
            grid_type = self.expression_result_type(grid_args[0])
            if self.map_type(grid_type) == "uint3":
                return None
            type_label = self.type_name_string(grid_type) or "unknown"
            return (
                "DispatchMesh grid argument must be a uint3-compatible vector"
                f"; got '{type_label}'"
            )

        for index, argument in enumerate(grid_args, start=1):
            argument_type = self.expression_result_type(argument)
            if self.is_scalar_integer_type(argument_type):
                continue
            type_label = self.type_name_string(argument_type) or "unknown"
            return (
                f"DispatchMesh grid component argument {index} must be a "
                f"scalar integer; got '{type_label}'"
            )
        return None

    def unsupported_metal_mesh_dispatch(self, reason):
        return f"/* unsupported Metal mesh dispatch: {reason} */"

    def metal_dispatch_mesh_payload_assignment(self, payload_expr):
        if self.current_metal_mesh_payload_parameter is None:
            return (
                "/* unsupported Metal mesh payload dispatch: "
                "DispatchMesh payload argument requires an object_data payload "
                "parameter */"
            )

        payload_type = self.expression_result_type(payload_expr)
        if payload_type is not None:
            mapped_payload_type = self.metal_mesh_payload_value_type(payload_type)
            if (
                self.current_metal_mesh_payload_type is not None
                and mapped_payload_type != self.current_metal_mesh_payload_type
            ):
                return (
                    "/* unsupported Metal mesh payload dispatch: "
                    f"payload argument type {mapped_payload_type} does not match "
                    f"{self.current_metal_mesh_payload_type} */"
                )

        payload_source_diagnostic = self.metal_dispatch_mesh_payload_source_diagnostic(
            payload_expr
        )
        if payload_source_diagnostic is not None:
            return payload_source_diagnostic

        payload_value = self.generate_expression(payload_expr)
        return f"{self.current_metal_mesh_payload_parameter} = {payload_value}"

    def metal_dispatch_mesh_payload_source_diagnostic(self, payload_expr):
        payload_name = self.assignment_target_root_name(payload_expr)
        if not payload_name:
            return (
                "/* unsupported Metal mesh payload dispatch: "
                "payload argument must be a threadgroup lvalue */"
            )

        payload_type = self.expression_result_type(payload_expr)
        if self.metal_type_is_pointer_like(payload_type):
            display_name = (
                self.assignment_target_display_name(payload_expr) or payload_name
            )
            return (
                "/* unsupported Metal mesh payload dispatch: payload argument "
                f"'{display_name}' has pointer or array type; DispatchMesh "
                "payload requires a threadgroup value lvalue */"
            )

        address_space = self.argument_address_space(payload_expr)
        if address_space == "threadgroup":
            return None

        source_address_space = address_space or "unknown"
        display_name = (
            self.assignment_target_access_display_name(payload_expr) or payload_name
        )
        return (
            "/* unsupported Metal mesh payload dispatch: payload argument "
            f"'{display_name}' uses {source_address_space} address space; "
            "DispatchMesh payload requires a threadgroup lvalue */"
        )

    def generate_buffer_call(self, func_name, args):
        if func_name == "buffer_load" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            return f"{buffer}[{index}]"
        if func_name == "buffer_store" and len(args) >= 3:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            return f"{buffer}[{index}] = {value}"
        if func_name == "buffer_append" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            value = self.generate_expression(args[1])
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return (
                    "/* unsupported Metal buffer append: requires append/consume "
                    "counter buffer */"
                )
            index = f"atomic_fetch_add_explicit({counter}, 1u, memory_order_relaxed)"
            return f"{buffer}[{index}] = {value}"
        if func_name == "buffer_consume" and args:
            buffer = self.generate_expression(args[0])
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return (
                    "0 /* unsupported Metal buffer consume: requires append/consume "
                    "counter buffer */"
                )
            index = (
                f"(atomic_fetch_sub_explicit({counter}, 1u, "
                "memory_order_relaxed) - 1u)"
            )
            return f"{buffer}[{index}]"
        if func_name == "buffer_dimensions" and args:
            length_expr = self.structured_buffer_length_reference(args[0])
            if length_expr is None:
                length_expr = (
                    "0 /* unsupported Metal buffer dimensions: requires explicit "
                    "length sidecar buffer */"
                )
            if len(args) >= 2:
                target = self.generate_expression(args[1])
                return f"{target} = {length_expr}"
            return length_expr
        return None

    def structured_buffer_length_reference(self, buffer_arg):
        length = self.structured_buffer_length_data_argument(buffer_arg)
        if length is None:
            return None
        return f"{length}[0]"

    def structured_buffer_length_data_argument(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_name = self.expression_name(array_expr)
            length_parameter = self.current_structured_buffer_length_parameters.get(
                array_name
            )
            if length_parameter is not None:
                index = self.generate_expression(index_expr)
                return f"{length_parameter}[{index}]"
            if self.structured_buffer_requires_length(array_name):
                index = self.generate_expression(index_expr)
                length_name = self.structured_buffer_length_parameter_name(array_name)
                return f"{length_name}[{index}]"

        arg_name = self.expression_name(arg)
        length_parameter = self.current_structured_buffer_length_parameters.get(
            arg_name
        )
        if length_parameter is not None:
            return length_parameter

        if self.structured_buffer_requires_length(arg_name):
            return self.structured_buffer_length_parameter_name(arg_name)
        return None

    def structured_buffer_counter_reference(self, buffer_arg):
        counter = self.structured_buffer_counter_data_argument(buffer_arg)
        if counter is None:
            return None
        return counter

    def structured_buffer_counter_data_argument(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_name = self.expression_name(array_expr)
            counter_parameter = self.current_structured_buffer_counter_parameters.get(
                array_name
            )
            if counter_parameter is not None:
                index = self.generate_expression(index_expr)
                return f"{counter_parameter}[{index}]"
            if self.global_structured_buffer_requires_counter(array_name):
                index = self.generate_expression(index_expr)
                counter_name = self.structured_buffer_counter_parameter_name(array_name)
                return f"{counter_name}[{index}]"

        arg_name = self.expression_name(arg)
        counter_parameter = self.current_structured_buffer_counter_parameters.get(
            arg_name
        )
        if counter_parameter is not None:
            return counter_parameter

        if self.global_structured_buffer_requires_counter(arg_name):
            return self.structured_buffer_counter_parameter_name(arg_name)
        return None

    def default_sampler_expression(self):
        return "sampler(mag_filter::linear, min_filter::linear)"

    def sampler_variable_names(self):
        return {
            sampler_variable.name for sampler_variable, _, _ in self.sampler_variables
        } | self.current_sampler_parameters

    def structured_buffer_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_structured_buffer_type(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_requires_counter(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_element_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return "uint"
        element_type = type_name.split("<", 1)[1][:-1].strip()
        return self.map_type(element_type)

    def structured_buffer_address_space(self, vtype):
        if self.structured_buffer_type_name(vtype) == "StructuredBuffer":
            return "const device"
        return "device"

    def format_structured_buffer_parameter(
        self, vtype, name, array_size=None, node=None
    ):
        element_type = self.structured_buffer_element_type(vtype)
        address_space = self.structured_buffer_address_space(vtype)
        memory_qualifiers = self.resource_memory_qualifier_prefix(node, vtype)
        pointer_type = f"{memory_qualifiers}{address_space} {element_type}*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type} {name}"

    def structured_buffer_length_resource_name(self, name):
        return f"{name}Length"

    def structured_buffer_length_parameter_name(self, name):
        return f"{name}Length"

    def format_structured_buffer_length_parameter(self, name, array_size=None):
        length_name = self.structured_buffer_length_parameter_name(name)
        pointer_type = "constant uint*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {length_name}"
        return f"{pointer_type} {length_name}"

    def structured_buffer_counter_resource_name(self, name):
        return f"{name}Counter"

    def structured_buffer_counter_parameter_name(self, name):
        return f"{name}Counter"

    def format_structured_buffer_counter_parameter(self, name, array_size=None):
        counter_name = self.structured_buffer_counter_parameter_name(name)
        pointer_type = "device atomic_uint*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {counter_name}"
        return f"{pointer_type} {counter_name}"

    def format_glsl_buffer_block_parameter(self, block, name, array_size=None):
        address_space = "const device" if block.get("readonly") else "device"
        pointer_type = f"{address_space} uchar*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type} {name}"

    def visible_function_table_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_visible_function_table_type(self, vtype):
        return self.visible_function_table_type_name(vtype) in {
            "visible_function_table",
            "visibleFunctionTable",
        }

    def visible_function_table_signature(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return "void()"

        payload_type = type_name.split("<", 1)[1][:-1].strip()
        if not payload_type or payload_type == "void":
            return "void()"
        return f"void(thread {self.map_type(payload_type)}&)"

    def map_visible_function_table_type(self, vtype):
        return f"visible_function_table<{self.visible_function_table_signature(vtype)}>"

    def format_visible_function_table_parameter(self, vtype, name, array_size=None):
        table_type = self.visible_function_table_parameter_type(vtype)
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{table_type}, {array_size}> {name}"
        return f"{table_type} {name}"

    def visible_function_table_parameter_type(self, vtype):
        type_name = str(vtype)
        if type_name.startswith("visible_function_table<"):
            signature = type_name.split("<", 1)[1][:-1].strip()
            if signature.startswith("void"):
                return type_name
        return self.map_visible_function_table_type(vtype)

    def intersection_function_table_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_intersection_function_table_type(self, vtype):
        return self.intersection_function_table_type_name(vtype) in {
            "intersection_function_table",
            "intersectionFunctionTable",
        }

    def map_intersection_function_table_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" in type_name and type_name.endswith(">"):
            tags = type_name.split("<", 1)[1][:-1].strip()
            return f"intersection_function_table<{tags}>"
        return "intersection_function_table<>"

    def format_intersection_function_table_parameter(
        self, vtype, name, array_size=None
    ):
        table_type = (
            vtype
            if str(vtype).startswith("intersection_function_table<")
            else self.map_intersection_function_table_type(vtype)
        )
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{table_type}, {array_size}> {name}"
        return f"{table_type} {name}"

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {"sampler", "comparison_sampler"}

    def is_acceleration_structure_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "accelerationStructureEXT",
            "RaytracingAccelerationStructure",
            "AccelerationStructure",
            "acceleration_structure",
            "instance_acceleration_structure",
            "primitive_acceleration_structure",
        }

    def is_resource_parameter_type(self, vtype):
        return (
            self.is_structured_buffer_type(vtype)
            or self.is_acceleration_structure_type(vtype)
            or self.is_visible_function_table_type(vtype)
            or self.is_intersection_function_table_type(vtype)
            or self.resource_base_type(vtype)
            in {
                "comparison_sampler",
                "sampler",
                "sampler1D",
                "sampler1DArray",
                "sampler2D",
                "sampler3D",
                "samplerCube",
                "sampler2DArray",
                "samplerCubeArray",
                "sampler2DMS",
                "sampler2DMSArray",
                "sampler2DShadow",
                "sampler2DArrayShadow",
                "samplerCubeShadow",
                "samplerCubeArrayShadow",
                "isampler1D",
                "isampler1DArray",
                "isampler2D",
                "isampler3D",
                "isamplerCube",
                "isampler2DArray",
                "isamplerCubeArray",
                "isampler2DMS",
                "isampler2DMSArray",
                "usampler1D",
                "usampler1DArray",
                "usampler2D",
                "usampler3D",
                "usamplerCube",
                "usampler2DArray",
                "usamplerCubeArray",
                "usampler2DMS",
                "usampler2DMSArray",
                "iimage1D",
                "iimage1DArray",
                "iimage2D",
                "iimage3D",
                "iimage2DArray",
                "iimage2DMS",
                "iimage2DMSArray",
                "uimage1D",
                "uimage1DArray",
                "uimage2D",
                "uimage3D",
                "uimage2DArray",
                "uimage2DMS",
                "uimage2DMSArray",
                "image1D",
                "image1DArray",
                "image2D",
                "image3D",
                "imageCube",
                "image2DArray",
                "image2DMS",
                "image2DMSArray",
            }
        )

    def is_bound_scalar_value_parameter(
        self, raw_param_type, node=None, shader_type=None
    ):
        if node is None or shader_type != "compute":
            return False
        if self.explicit_buffer_binding_index(node) is None:
            return False
        qualifiers = self.parameter_qualifier_names(node)
        if not qualifiers & {"uniform", "constant"}:
            return False
        if self.is_resource_parameter_type(raw_param_type):
            return False
        if self.is_raw_buffer_parameter_type(raw_param_type, node):
            return False
        type_name = self.type_name_string(raw_param_type)
        return is_numeric_scalar_type_name(type_name, self.map_type)

    def is_texture_or_image_resource_type(self, vtype):
        return (
            self.is_resource_parameter_type(vtype)
            and not self.is_sampler_type(vtype)
            and not self.is_structured_buffer_type(vtype)
            and not self.is_acceleration_structure_type(vtype)
        )

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        return is_integer_coordinate_type_name(base_type, mapped_type)

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = "_ms" in texture_type
        is_storage_image = "access::" in texture_type

        coordinate_dimension = None
        if texture_type and "cube" not in texture_type:
            if texture_type.startswith("texture1d_array<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture1d<"):
                coordinate_dimension = 1
            elif texture_type.startswith("texture2d_ms_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("texture2d_ms<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture2d_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("texture2d<"):
                coordinate_dimension = 2
            elif texture_type.startswith("depth2d_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("depth2d<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture3d<"):
                coordinate_dimension = 3

        texel_fetch_offset_dimension = None
        if texture_type and "cube" not in texture_type and not is_multisample:
            if texture_type.startswith("texture1d_array<"):
                texel_fetch_offset_dimension = 1
            elif texture_type.startswith("texture1d<"):
                texel_fetch_offset_dimension = 1
            elif texture_type.startswith("texture2d_array<"):
                texel_fetch_offset_dimension = 2
            elif texture_type.startswith("texture2d<"):
                texel_fetch_offset_dimension = 2
            elif texture_type.startswith("texture3d<"):
                texel_fetch_offset_dimension = 3

        sample_offset_dimension = None
        if texture_type.startswith("texture2d_array<"):
            sample_offset_dimension = 2
        elif texture_type.startswith("texture2d<"):
            sample_offset_dimension = 2
        elif texture_type.startswith("texture3d<"):
            sample_offset_dimension = 3
        if not sampling["sample_offset"]:
            sample_offset_dimension = None

        gradient_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            if texture_type.startswith(("texture2d_array<", "depth2d_array<")):
                gradient_dimension = 2
            elif texture_type.startswith(("texture2d<", "depth2d<")):
                gradient_dimension = 2
            elif texture_type.startswith("texture3d<"):
                gradient_dimension = 3
            elif texture_type.startswith(("texturecube_array<", "depthcube_array<")):
                gradient_dimension = 3
            elif texture_type.startswith(("texturecube<", "depthcube<")):
                gradient_dimension = 3

        query_lod_coordinate_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            if texture_type.startswith(("texture2d_array<", "depth2d_array<")):
                query_lod_coordinate_dimension = 3
            elif texture_type.startswith(("texture2d<", "depth2d<")):
                query_lod_coordinate_dimension = 2
            elif texture_type.startswith("texture3d<"):
                query_lod_coordinate_dimension = 3
            elif texture_type.startswith(("texturecube_array<", "depthcube_array<")):
                query_lod_coordinate_dimension = 4
            elif texture_type.startswith(("texturecube<", "depthcube<")):
                query_lod_coordinate_dimension = 3

        return texture_resource_dimension_descriptor(
            texture_type,
            sampling,
            coordinate_dimension=coordinate_dimension,
            offset_dimension=sample_offset_dimension,
            sample_offset_dimension=sample_offset_dimension,
            texel_fetch_offset_dimension=texel_fetch_offset_dimension,
            gradient_dimension=gradient_dimension,
            query_lod_coordinate_dimension=query_lod_coordinate_dimension,
            is_multisample=is_multisample,
        )

    def resource_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)["coordinate_dimension"]

    def resource_offset_dimension(self, func_name, texture_type):
        descriptor = self.texture_dimension_descriptor(texture_type)
        key = texture_resource_offset_dimension_key(
            func_name, collapse_compare_offsets=True
        )
        return descriptor[key]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

    def parameter_attribute(self, raw_param_type, semantic, shader_type, node=None):
        ray_semantic_role = self.metal_ray_semantic_role(node, shader_type)
        if shader_type == "ray_generation" and ray_semantic_role == "payload":
            return ""
        if self.is_metal_ray_stage(shader_type) and ray_semantic_role == "payload":
            return " [[payload]]"
        if (
            self.is_metal_ray_stage(shader_type)
            and ray_semantic_role == "hit_attribute"
        ):
            return ""
        if (
            shader_type in {"ray_callable", "callable"}
            and ray_semantic_role == "callable_data"
        ):
            return ""
        if self.is_metal_tessellation_helper_semantic(semantic) and shader_type in {
            None,
            "tessellation_control",
            "tessellation_evaluation",
        }:
            return ""
        if semantic:
            return self.map_semantic(semantic)
        if shader_type in {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "ray_generation",
            "ray_intersection",
            "intersection",
            "object",
            "task",
            "amplification",
            "mesh",
        }:
            resource_attr = self.resource_parameter_attribute(raw_param_type, node)
            if resource_attr:
                return resource_attr
        if self.is_resource_parameter_type(raw_param_type):
            return ""
        if shader_type in {"vertex", "fragment"}:
            return " [[stage_in]]"
        return ""

    def parameter_resource_binding_metadata(self, raw_param_type, node=None):
        if node is None:
            return None
        if self.is_sampler_type(raw_param_type):
            namespace = "sampler"
            attribute_names = {"binding", "sampler"}
            prefixes = ("s",)
        elif self.is_visible_function_table_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_intersection_function_table_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_acceleration_structure_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_structured_buffer_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_texture_or_image_resource_type(raw_param_type):
            namespace = "texture"
            attribute_names = {"binding", "texture"}
            prefixes = ("t", "u")
        elif self.is_bound_scalar_value_parameter(raw_param_type, node, "compute"):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_raw_buffer_parameter_type(raw_param_type, node):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        else:
            return None

        binding = self.explicit_resource_binding_index(node, attribute_names, prefixes)
        if binding is None:
            return None
        return (
            namespace,
            binding,
            self.parameter_resource_binding_count(raw_param_type, node),
            getattr(node, "name", "<anonymous>"),
        )

    def parameter_resource_binding_count(self, raw_param_type, node=None):
        array_size = None
        if self.is_structured_buffer_type(raw_param_type):
            array_size = self.structured_buffer_parameter_array_size(
                raw_param_type, node
            )
        else:
            array_type = self.resource_array_parameter(raw_param_type, node)
            if array_type is not None:
                _, array_size = array_type
        return self.resource_array_count(array_size)

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return (
                self.type_name_string(parameter.param_type)
                if getattr(parameter.param_type, "generic_args", None)
                else parameter.param_type
            )
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "float"

    def validate_stage_parameter_resource_bindings(self, parameters, func_name=None):
        used_bindings = {}
        self.reserve_global_parameter_resource_bindings(used_bindings, func_name)
        for parameter in parameters or []:
            metadata = self.parameter_resource_binding_metadata(
                self.parameter_raw_type(parameter), parameter
            )
            if metadata is None:
                continue
            namespace, binding, resource_count, name = metadata
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                namespace,
                binding,
                resource_count,
                name,
            )

    def reserve_global_parameter_resource_bindings(self, used_bindings, func_name=None):
        dependencies = (
            self.function_global_resource_dependencies.get(func_name, set())
            if func_name
            else None
        )
        for cbuffer in self.cbuffer_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                self.cbuffer_binding_indices.get(id(cbuffer), 0),
                1,
                getattr(cbuffer, "name", "<anonymous>"),
            )
        for texture_variable, binding, _, _ in self.texture_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "texture",
                binding,
                self.global_resource_shape(texture_variable)[1],
                getattr(texture_variable, "name", "<anonymous>"),
            )
        for (
            acceleration_structure_variable,
            binding,
            _,
            _,
        ) in self.acceleration_structure_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(acceleration_structure_variable)[1],
                getattr(acceleration_structure_variable, "name", "<anonymous>"),
            )
        for (
            visible_function_table_variable,
            binding,
            _,
            _,
        ) in self.visible_function_table_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(visible_function_table_variable)[1],
                getattr(visible_function_table_variable, "name", "<anonymous>"),
            )
        for (
            intersection_function_table_variable,
            binding,
            _,
            _,
        ) in self.intersection_function_table_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(intersection_function_table_variable)[1],
                getattr(intersection_function_table_variable, "name", "<anonymous>"),
            )
        for buffer_variable, binding, _, _ in self.structured_buffer_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                getattr(buffer_variable, "name", "<anonymous>"),
            )
        for (
            buffer_variable,
            binding,
            _,
            array_size,
            _,
        ) in self.metal_buffer_resource_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.metal_buffer_resource_binding_count(buffer_variable, array_size),
                getattr(buffer_variable, "name", "<anonymous>"),
            )
        for buffer_variable, binding, _, _ in self.structured_buffer_length_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                self.structured_buffer_length_resource_name(
                    getattr(buffer_variable, "name", "<anonymous>")
                ),
            )
        for buffer_variable, binding, _, _ in self.structured_buffer_counter_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                self.structured_buffer_counter_resource_name(
                    getattr(buffer_variable, "name", "<anonymous>")
                ),
            )
        for buffer_variable, binding, _, _ in self.glsl_buffer_block_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                getattr(buffer_variable, "name", "<anonymous>"),
            )
        for sampler_variable, binding, array_size in self.sampler_variables:
            sampler_name = getattr(sampler_variable, "name", None)
            if dependencies is not None and sampler_name not in dependencies:
                continue
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "sampler",
                binding,
                self.resource_array_count(array_size),
                sampler_name or "<anonymous>",
            )

    def resource_parameter_attribute(self, raw_param_type, node=None):
        if node is None:
            return ""
        if self.is_sampler_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "sampler"}, ("s",)
            )
            return f" [[sampler({binding})]]" if binding is not None else ""
        if self.is_visible_function_table_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_intersection_function_table_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_acceleration_structure_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_structured_buffer_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_resource_parameter_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "texture"}, ("t", "u")
            )
            return f" [[texture({binding})]]" if binding is not None else ""
        if self.is_bound_scalar_value_parameter(raw_param_type, node, "compute"):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_raw_buffer_parameter_type(raw_param_type, node):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        return ""

    def format_parameter_declaration(
        self, raw_param_type, mapped_type, name, node=None, shader_type=None
    ):
        self.validate_metal_ray_resource_parameter_value_type(raw_param_type, name)

        geometry_stream_declaration = self.format_metal_geometry_stream_parameter(
            raw_param_type,
            name,
        )
        if geometry_stream_declaration is not None:
            return geometry_stream_declaration

        tessellation_patch_declaration = self.format_metal_tessellation_patch_parameter(
            raw_param_type, name
        )
        if tessellation_patch_declaration is not None:
            return tessellation_patch_declaration

        ray_payload_declaration = self.metal_ray_payload_parameter_declaration(
            mapped_type, name, node, shader_type
        )
        if ray_payload_declaration is not None:
            return ray_payload_declaration

        mesh_payload_declaration = self.metal_mesh_payload_parameter_declaration(
            mapped_type, name, node, shader_type
        )
        if mesh_payload_declaration is not None:
            return mesh_payload_declaration

        lowered_block = self.current_glsl_buffer_block_parameters.get(name)
        if lowered_block is not None:
            array_size = self.glsl_buffer_block_parameter_array_size(
                raw_param_type, node
            )
            return self.format_glsl_buffer_block_parameter(
                lowered_block, name, array_size
            )
        if self.is_structured_buffer_type(raw_param_type):
            array_size = self.structured_buffer_parameter_array_size(
                raw_param_type, node
            )
            return self.format_structured_buffer_parameter(
                raw_param_type, name, array_size, node
            )
        if self.is_visible_function_table_type(raw_param_type):
            return self.format_visible_function_table_parameter(raw_param_type, name)
        if self.is_intersection_function_table_type(raw_param_type):
            return self.format_intersection_function_table_parameter(
                raw_param_type, name
            )
        if self.is_bound_scalar_value_parameter(raw_param_type, node, shader_type):
            return f"constant {mapped_type}& {name}"
        address_space_declaration = self.format_address_space_parameter_declaration(
            raw_param_type, mapped_type, name, node, shader_type
        )
        if address_space_declaration is not None:
            return address_space_declaration
        array_type = self.resource_array_parameter(raw_param_type, node)
        if array_type is not None:
            resource_type, array_size = array_type
            return self.format_resource_parameter(resource_type, name, array_size)
        return format_c_style_array_declaration(mapped_type, name)

    def validate_metal_ray_resource_parameter_value_type(self, raw_param_type, name):
        resource_kind = self.metal_ray_resource_pointer_or_reference_kind(
            raw_param_type
        )
        if resource_kind is None:
            return

        type_name = self.type_name_string(raw_param_type) or str(raw_param_type)
        raise ValueError(
            f"Metal ray tracing resource parameter '{name}' has pointer or "
            f"reference type '{type_name}' for {resource_kind}; Metal ray "
            "tracing resources must be passed by value without address-space "
            "qualifiers"
        )

    def metal_ray_resource_pointer_or_reference_kind(self, raw_param_type):
        if isinstance(raw_param_type, PointerType):
            base_type = raw_param_type.pointee_type
        elif isinstance(raw_param_type, ReferenceType):
            base_type = raw_param_type.referenced_type
        else:
            type_name = self.type_name_string(raw_param_type)
            if not type_name:
                return None
            type_name = str(type_name).strip()
            if not (type_name.endswith("*") or type_name.endswith("&")):
                return None
            base_type = type_name[:-1].strip()

        if self.is_acceleration_structure_type(base_type):
            return "acceleration_structure"
        if self.is_visible_function_table_type(base_type):
            return "visible_function_table"
        if self.is_intersection_function_table_type(base_type):
            return "intersection_function_table"
        return None

    def format_address_space_parameter_declaration(
        self, raw_param_type, mapped_type, name, node=None, shader_type=None
    ):
        if isinstance(raw_param_type, PointerType):
            address_space = self.effective_parameter_address_space(
                raw_param_type, node, shader_type
            )
            address_space = self.readonly_qualified_address_space(address_space, node)
            pointee_type = self.map_resource_type_with_format(
                raw_param_type.pointee_type, node
            )
            memory_qualifiers = self.resource_memory_qualifier_prefix(
                node, raw_param_type
            )
            return f"{memory_qualifiers}{address_space} {pointee_type}* {name}"

        if isinstance(raw_param_type, ReferenceType):
            address_space = self.effective_parameter_address_space(
                raw_param_type, node, shader_type
            )
            address_space = self.readonly_qualified_address_space(address_space, node)
            referenced_type = self.map_resource_type_with_format(
                raw_param_type.referenced_type, node
            )
            memory_qualifiers = self.resource_memory_qualifier_prefix(
                node, raw_param_type
            )
            return f"{memory_qualifiers}{address_space} {referenced_type}& {name}"

        if self.is_array_type_node(raw_param_type):
            if self.resource_array_parameter(raw_param_type, node) is not None:
                return None
            binding = self.explicit_buffer_binding_index(node)
            qualifiers = self.parameter_qualifier_names(node)
            address_space = self.effective_parameter_address_space(
                raw_param_type,
                node,
                shader_type,
                default_for_stage_binding=binding is not None
                or bool(qualifiers & {"const", "readonly"}),
            )
            if address_space is None:
                return None
            address_space = self.readonly_qualified_address_space(address_space, node)
            if binding is not None and shader_type in {
                "vertex",
                "fragment",
                "compute",
                "ray_generation",
            }:
                element_type = self.map_resource_type_with_format(
                    raw_param_type.element_type, node
                )
                memory_qualifiers = self.resource_memory_qualifier_prefix(
                    node, raw_param_type
                )
                return f"{memory_qualifiers}{address_space} {element_type}* {name}"
            declaration = format_c_style_array_declaration(mapped_type, name)
            memory_qualifiers = self.resource_memory_qualifier_prefix(
                node, raw_param_type
            )
            return f"{memory_qualifiers}{address_space} {declaration}"

        qualifiers = self.parameter_qualifier_names(node)
        address_space = self.effective_parameter_address_space(
            raw_param_type,
            node,
            shader_type,
            default_for_stage_binding=bool(qualifiers & {"out", "inout"}),
        )
        if address_space is not None:
            address_space = self.readonly_qualified_address_space(
                address_space,
                node,
            )
            memory_qualifiers = self.resource_memory_qualifier_prefix(
                node, raw_param_type
            )
            return f"{memory_qualifiers}{address_space} {mapped_type}& {name}"

        return None

    def effective_parameter_address_space(
        self,
        raw_param_type,
        node=None,
        shader_type=None,
        default_for_stage_binding=True,
    ):
        default_space = None
        if default_for_stage_binding or isinstance(
            raw_param_type, (PointerType, ReferenceType)
        ):
            default_space = self.default_parameter_address_space(
                raw_param_type, node, shader_type
            )
        return self.parameter_address_space(node, default=default_space)

    def default_parameter_address_space(
        self, raw_param_type, node=None, shader_type=None
    ):
        if shader_type in {"vertex", "fragment", "compute", "ray_generation"}:
            binding = self.explicit_buffer_binding_index(node)
            if binding is not None:
                if isinstance(raw_param_type, ReferenceType):
                    return "constant"
                return "device"
        return "thread"

    def explicit_buffer_binding_index(self, node):
        return self.explicit_resource_binding_index(
            node, {"binding", "buffer"}, ("b", "u", "t")
        )

    def parameter_address_space(self, node=None, default=None):
        qualifiers = self.parameter_qualifier_names(node)
        address_spaces = []
        if "constant" in qualifiers:
            address_spaces.append("constant")
        if qualifiers & {"object_data", "objectdata"}:
            address_spaces.append("object_data")
        if qualifiers & {"ray_data", "raydata"}:
            address_spaces.append("ray_data")
        if qualifiers & {"device", "global", "storage"}:
            address_spaces.append("device")
        if "threadgroup_imageblock" in qualifiers:
            address_spaces.append("threadgroup_imageblock")
        if qualifiers & {"threadgroup", "workgroup", "shared", "groupshared"}:
            address_spaces.append("threadgroup")
        if qualifiers & {"thread", "function", "local", "private"}:
            address_spaces.append("thread")

        address_spaces = list(dict.fromkeys(address_spaces))
        if len(address_spaces) > 1:
            name = getattr(node, "name", "<anonymous>")
            raise ValueError(
                f"Metal parameter '{name}' has conflicting address-space qualifiers: "
                f"{', '.join(sorted(qualifiers))}"
            )
        if address_spaces:
            return address_spaces[0]
        return default

    def parameter_qualifier_names(self, node=None):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        qualifiers.update(
            str(qualifier).lower()
            for qualifier in getattr(node, "resource_qualifiers", []) or []
        )
        qualifiers.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        )
        raw_type = getattr(node, "param_type", getattr(node, "var_type", None))
        contract_type = raw_type
        while self.is_array_type_node(contract_type):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            qualifiers.add(str(getattr(contract_type, "address_space", "")).lower())
            qualifiers.add(str(getattr(contract_type, "access_mode", "")).lower())
        qualifier_aliases = {
            "read": "readonly",
            "write": "writeonly",
            "read_write": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        qualifiers.update(
            alias
            for qualifier, alias in qualifier_aliases.items()
            if qualifier in qualifiers
        )
        qualifiers.discard("")
        return qualifiers

    def resource_memory_qualifier_contracts(self, node=None, raw_type=None):
        """Return ordered ``(kind, scope)`` resource-memory contracts."""
        values = []
        contract_type = raw_type
        while self.is_array_type_node(contract_type):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            values.extend(getattr(contract_type, "resource_qualifiers", []) or [])
        values.extend(getattr(node, "resource_qualifiers", []) or [])
        values.extend(getattr(node, "qualifiers", []) or [])

        contracts = []
        for value in values:
            kind = getattr(value, "kind", None)
            scope = getattr(value, "scope", None)
            text = str(value).lower()
            if kind is None:
                match = re.fullmatch(
                    r"(?P<kind>volatile|coherent)(?:\((?P<scope>[a-z_][a-z0-9_]*)\))?",
                    text,
                )
                if match is None:
                    continue
                kind = match.group("kind")
                scope = match.group("scope")
            else:
                kind = str(kind).lower()
                scope = str(scope).lower() if scope is not None else None
            if kind not in {"volatile", "coherent"}:
                continue
            contract = (kind, scope)
            if contract not in contracts:
                contracts.append(contract)

        for attribute in getattr(node, "attributes", []) or []:
            kind = str(getattr(attribute, "name", "")).lower()
            if kind not in {"volatile", "coherent"}:
                continue
            arguments = getattr(attribute, "arguments", []) or []
            scope = self.attribute_value_to_string(arguments[0]) if arguments else None
            contract = (kind, str(scope).lower() if scope is not None else None)
            if contract not in contracts:
                contracts.append(contract)
        return contracts

    def resource_memory_qualifier_prefix(self, node=None, raw_type=None):
        rendered = []
        for kind, scope in self.resource_memory_qualifier_contracts(node, raw_type):
            if kind == "coherent" and scope is not None:
                if scope not in {"threadgroup", "device", "system"}:
                    name = getattr(node, "name", "<anonymous>")
                    raise ValueError(
                        f"Metal resource '{name}' has unsupported coherence scope "
                        f"'{scope}'"
                    )
                rendered.append(f"coherent({scope})")
            else:
                rendered.append(kind)
        return f"{' '.join(rendered)} " if rendered else ""

    def normalized_address_space(self, address_space):
        if address_space is None:
            return None
        address_space = str(address_space).strip()
        if address_space.startswith("const "):
            address_space = address_space[len("const ") :].strip()
        return address_space or None

    def readonly_qualified_address_space(self, address_space, node=None):
        qualifiers = self.parameter_qualifier_names(node)
        if address_space == "device" and qualifiers & {"in", "readonly"}:
            return "const device"
        if qualifiers & {"const", "in"} and address_space not in {None, "constant"}:
            return f"const {address_space}"
        return address_space

    def parameter_variable_address_space(
        self, raw_param_type, node=None, shader_type=None
    ):
        if (
            self.is_array_type_node(raw_param_type)
            and self.resource_array_parameter(raw_param_type, node) is not None
        ):
            return None
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return None
        return self.normalized_address_space(
            self.effective_parameter_address_space(
                raw_param_type,
                node,
                shader_type,
                default_for_stage_binding=self.explicit_buffer_binding_index(node)
                is not None,
            )
        )

    def readonly_metal_parameter_reason(
        self, raw_param_type, node=None, shader_type=None
    ):
        if self.is_metal_mesh_payload_parameter(shader_type, node):
            return None
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return None
        qualifiers = self.parameter_qualifier_names(node)
        if "const" in qualifiers:
            return "const-qualified"
        if "in" in qualifiers:
            return "input-only"
        if "constant" in qualifiers:
            return "constant address space"
        if "readonly" in qualifiers and not self.is_raw_buffer_parameter_type(
            raw_param_type, node
        ):
            return "readonly"
        return None

    def is_mutable_metal_parameter(self, raw_param_type, node=None):
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return False
        if self.parameter_qualifier_names(node) & {
            "const",
            "constant",
            "in",
            "readonly",
        }:
            return False
        return self.readonly_metal_parameter_reason(raw_param_type, node) is None

    def is_readonly_raw_buffer_parameter(
        self, raw_param_type, node=None, shader_type=None
    ):
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return False
        qualifiers = self.parameter_qualifier_names(node)
        if qualifiers & {"in", "readonly"}:
            return True
        address_space = self.effective_parameter_address_space(
            raw_param_type,
            node,
            shader_type,
            default_for_stage_binding=self.explicit_buffer_binding_index(node)
            is not None,
        )
        return address_space == "constant"

    def is_array_type_node(self, vtype):
        return (
            hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1
        )

    def is_raw_buffer_parameter_type(self, raw_param_type, node=None):
        if isinstance(
            raw_param_type, (PointerType, ReferenceType)
        ) or self.is_array_type_node(raw_param_type):
            address_space = self.parameter_address_space(node)
            return address_space in {"device", "constant"} or (
                address_space is None
                and self.explicit_buffer_binding_index(node) is not None
            )
        return False

    def is_mutable_raw_buffer_parameter(self, raw_param_type, node=None):
        if not self.is_raw_buffer_parameter_type(raw_param_type, node):
            return False
        return not self.is_readonly_raw_buffer_parameter(raw_param_type, node)

    def readonly_raw_buffer_call_diagnostic(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            arg_name = self.assignment_target_root_name(arg)
            if arg_name not in self.current_readonly_raw_buffer_parameters:
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            if not self.is_mutable_raw_buffer_parameter(raw_param_type, parameter):
                continue
            parameter_name = getattr(parameter, "name", f"arg{index}")
            return (
                "/* unsupported Metal raw buffer call: readonly buffer "
                f"'{arg_name}' cannot be passed to mutable parameter "
                f"'{parameter_name}' of '{func_name}' */"
            )
        return None

    def readonly_metal_parameter_call_diagnostic(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            arg_name = self.assignment_target_root_name(arg)
            if arg_name not in self.current_readonly_metal_parameters:
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            if not self.is_mutable_metal_parameter(raw_param_type, parameter):
                continue
            parameter_name = getattr(parameter, "name", f"arg{index}")
            return (
                "/* unsupported Metal parameter call: readonly parameter "
                f"'{arg_name}' cannot be passed to mutable parameter "
                f"'{parameter_name}' of '{func_name}' */"
            )
        return None

    def readonly_metal_mesh_payload_call_diagnostic(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            arg_name = self.readonly_metal_mesh_payload_key(arg)
            if arg_name is None:
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            expected_address_space = self.parameter_variable_address_space(
                raw_param_type, parameter
            )
            if expected_address_space != "object_data":
                continue
            if not self.is_mutable_metal_parameter(raw_param_type, parameter):
                continue
            parameter_name = getattr(parameter, "name", f"arg{index}")
            reason = self.current_readonly_metal_mesh_payload_reasons.get(
                arg_name, "const object_data"
            )
            reason_text = self.readonly_metal_mesh_payload_reason_text(reason)
            diagnostic = (
                "/* unsupported Metal mesh payload call: mesh payload "
                f"'{arg_name}' is {reason_text} and cannot be passed to "
                f"mutable parameter '{parameter_name}' of '{func_name}' */"
            )
            return_type = self.function_return_types.get(func_name)
            if self.map_type(return_type) == "void":
                return diagnostic
            return f"{self.diagnostic_zero_value_for_type(return_type)} {diagnostic}"
        return None

    def argument_address_space(self, arg):
        if isinstance(arg, TernaryOpNode):
            if self.argument_address_space_conflict(arg) is not None:
                return None
            true_space = self.argument_address_space(getattr(arg, "true_expr", None))
            false_space = self.argument_address_space(getattr(arg, "false_expr", None))
            if (
                true_space is not None
                and false_space is not None
                and true_space == false_space
            ):
                return true_space
            return None
        member_address_space = self.address_space_qualified_member_address_space(arg)
        if member_address_space is not None:
            return member_address_space
        arg_name = self.assignment_target_root_name(arg)
        if arg_name in self.current_address_space_variables:
            return self.current_address_space_variables[arg_name]
        if arg_name in self.local_variable_types:
            return "thread"
        return None

    def address_space_qualified_member_address_space(self, expr):
        if isinstance(expr, UnaryOpNode) and getattr(expr, "operator", None) == "&":
            return self.address_space_qualified_member_address_space(
                getattr(expr, "operand", None)
            )
        if isinstance(expr, BinaryOpNode) and getattr(expr, "operator", None) in {
            "+",
            "-",
        }:
            return self.address_space_qualified_member_address_space(
                getattr(expr, "left", None)
            ) or self.address_space_qualified_member_address_space(
                getattr(expr, "right", None)
            )
        if isinstance(expr, ArrayAccessNode):
            return self.address_space_qualified_member_address_space(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            object_type = self.expression_result_type(object_expr)
            object_type = self.member_lookup_type_name(object_type)
            return self.struct_member_address_spaces.get(
                self.type_name_string(object_type), {}
            ).get(str(getattr(expr, "member", "")))
        if isinstance(expr, PointerAccessNode):
            object_expr = getattr(expr, "pointer_expr", None)
            object_type = self.expression_result_type(object_expr)
            object_type = self.member_lookup_type_name(object_type)
            return self.struct_member_address_spaces.get(
                self.type_name_string(object_type), {}
            ).get(str(getattr(expr, "member", "")))
        return None

    def argument_address_space_conflict(self, arg):
        if arg is None:
            return None
        if isinstance(arg, TernaryOpNode):
            true_expr = getattr(arg, "true_expr", None)
            false_expr = getattr(arg, "false_expr", None)
            true_conflict = self.argument_address_space_conflict(true_expr)
            if true_conflict is not None:
                return true_conflict
            false_conflict = self.argument_address_space_conflict(false_expr)
            if false_conflict is not None:
                return false_conflict
            true_space = self.argument_address_space(true_expr)
            false_space = self.argument_address_space(false_expr)
            if (
                true_space is not None
                and false_space is not None
                and true_space != false_space
            ):
                return (
                    true_space,
                    false_space,
                    self.assignment_target_display_name(true_expr) or "<expr>",
                    self.assignment_target_display_name(false_expr) or "<expr>",
                )
            return None
        if isinstance(arg, UnaryOpNode) and getattr(arg, "operator", None) in {
            "&",
            "*",
        }:
            return self.argument_address_space_conflict(getattr(arg, "operand", None))
        if isinstance(arg, BinaryOpNode) and getattr(arg, "operator", None) in {
            "+",
            "-",
        }:
            return self.argument_address_space_conflict(
                getattr(arg, "left", None)
            ) or self.argument_address_space_conflict(getattr(arg, "right", None))
        if isinstance(arg, ArrayAccessNode):
            return self.argument_address_space_conflict(
                getattr(arg, "array", getattr(arg, "array_expr", None))
            )
        return None

    def address_space_conflict_description(self, conflict):
        true_space, false_space, true_name, false_name = conflict
        return (
            f"branches '{true_name}' ({true_space}) and "
            f"'{false_name}' ({false_space})"
        )

    def address_space_call_diagnostic(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            expected_address_space = self.parameter_variable_address_space(
                raw_param_type, parameter
            )
            if expected_address_space is None:
                continue
            address_space_conflict = self.argument_address_space_conflict(arg)
            if address_space_conflict is not None:
                arg_name = self.assignment_target_display_name(arg) or "<expr>"
                parameter_name = getattr(parameter, "name", f"arg{index}")
                diagnostic = (
                    "/* unsupported Metal address-space call: argument "
                    f"'{arg_name}' mixes "
                    f"{self.address_space_conflict_description(address_space_conflict)} "
                    f"but parameter '{parameter_name}' of '{func_name}' requires "
                    f"{expected_address_space} */"
                )
                return_type = self.function_return_types.get(func_name)
                if self.map_type(return_type) == "void":
                    return diagnostic
                return (
                    f"{self.diagnostic_zero_value_for_type(return_type)} {diagnostic}"
                )
            actual_address_space = self.argument_address_space(arg)
            if (
                actual_address_space is None
                or actual_address_space == expected_address_space
            ):
                continue
            arg_name = self.assignment_target_display_name(arg)
            parameter_name = getattr(parameter, "name", f"arg{index}")
            diagnostic = (
                "/* unsupported Metal address-space call: argument "
                f"'{arg_name}' uses {actual_address_space} address space but "
                f"parameter '{parameter_name}' of '{func_name}' requires "
                f"{expected_address_space} */"
            )
            return_type = self.function_return_types.get(func_name)
            if self.map_type(return_type) == "void":
                return diagnostic
            return f"{self.diagnostic_zero_value_for_type(return_type)} {diagnostic}"
        return None

    def pointer_pointee_type_name(self, vtype):
        if isinstance(vtype, PointerType):
            return self.type_name_string(vtype.pointee_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("*") else None

    def reference_referent_type_name(self, vtype):
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("&") else None

    def member_lookup_type_name(self, vtype):
        return (
            self.pointer_pointee_type_name(vtype)
            or self.reference_referent_type_name(vtype)
            or vtype
        )

    def member_access_uses_pointer_operator(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return False
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        object_type = self.expression_result_type(object_expr)
        return self.pointer_pointee_type_name(object_type) is not None

    def collect_struct_member_address_spaces(self, structs):
        member_address_spaces = {}
        for struct in structs or []:
            struct_name = getattr(struct, "name", None)
            members = getattr(struct, "members", None)
            if not struct_name or members is None:
                continue
            for member in members:
                member_name = getattr(member, "name", None)
                raw_member_type = getattr(member, "member_type", None)
                if not member_name or raw_member_type is None:
                    continue
                if not (
                    isinstance(raw_member_type, (PointerType, ReferenceType))
                    or self.is_array_type_node(raw_member_type)
                ):
                    continue
                address_space = self.parameter_variable_address_space(
                    raw_member_type, member
                )
                if address_space is not None:
                    member_address_spaces.setdefault(struct_name, {})[
                        member_name
                    ] = address_space
        return member_address_spaces

    def assignment_target_root_name(self, target):
        if isinstance(target, UnaryOpNode) and getattr(target, "operator", None) in {
            "&",
            "*",
        }:
            return self.assignment_target_root_name(getattr(target, "operand", None))
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_root_name(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
        if isinstance(target, PointerAccessNode):
            return self.assignment_target_root_name(
                getattr(target, "pointer_expr", None)
            )
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_root_name(
                getattr(target, "array", getattr(target, "array_expr", None))
            )
        if isinstance(target, BinaryOpNode) and getattr(target, "operator", None) in {
            "+",
            "-",
        }:
            return self.assignment_target_root_name(
                getattr(target, "left", None)
            ) or self.assignment_target_root_name(getattr(target, "right", None))
        return self.expression_name(target)

    def assignment_target_display_name(self, target):
        if isinstance(target, UnaryOpNode):
            operand_name = self.assignment_target_display_name(
                getattr(target, "operand", None)
            )
            if getattr(target, "operator", None) == "&":
                return operand_name
            if getattr(target, "operator", None) == "*":
                return f"*{operand_name}" if operand_name else None
        if isinstance(target, BinaryOpNode) and getattr(target, "operator", None) in {
            "+",
            "-",
        }:
            return self.assignment_target_display_name(
                getattr(target, "left", None)
            ) or self.assignment_target_display_name(getattr(target, "right", None))
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_display_name(
                getattr(target, "array", getattr(target, "array_expr", None))
            )
        if isinstance(target, MemberAccessNode):
            object_name = self.assignment_target_display_name(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}.{member_name}" if object_name else member_name
        if isinstance(target, PointerAccessNode):
            object_name = self.assignment_target_display_name(
                getattr(target, "pointer_expr", None)
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}->{member_name}" if object_name else member_name
        return self.expression_name(target)

    def assignment_target_access_display_name(self, target):
        if isinstance(target, UnaryOpNode):
            operand_name = self.assignment_target_access_display_name(
                getattr(target, "operand", None)
            )
            if getattr(target, "operator", None) == "&":
                return operand_name
            if getattr(target, "operator", None) == "*":
                return f"*{operand_name}" if operand_name else None
        if isinstance(target, BinaryOpNode) and getattr(target, "operator", None) in {
            "+",
            "-",
        }:
            return self.assignment_target_access_display_name(
                getattr(target, "left", None)
            ) or self.assignment_target_access_display_name(
                getattr(target, "right", None)
            )
        if isinstance(target, ArrayAccessNode):
            object_name = self.assignment_target_access_display_name(
                getattr(target, "array", getattr(target, "array_expr", None))
            )
            index_expr = getattr(target, "index", getattr(target, "index_expr", None))
            index_name = (
                self.safe_expression_to_string(index_expr)
                if index_expr is not None
                else ""
            )
            return f"{object_name}[{index_name}]" if object_name else None
        if isinstance(target, MemberAccessNode):
            object_name = self.assignment_target_access_display_name(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}.{member_name}" if object_name else member_name
        if isinstance(target, PointerAccessNode):
            object_name = self.assignment_target_access_display_name(
                getattr(target, "pointer_expr", None)
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}->{member_name}" if object_name else member_name
        return self.expression_name(target)

    def readonly_metal_mesh_payload_key(self, target):
        display_name = self.assignment_target_display_name(target)
        if display_name:
            candidates = [display_name]
            for separator in (".", "->", "["):
                if separator not in display_name:
                    continue
                parts = display_name.split(separator)
                prefix = parts[0]
                for part in parts[1:-1]:
                    prefix = f"{prefix}{separator}{part}"
                    candidates.append(prefix)
            for candidate in sorted(candidates, key=len, reverse=True):
                if candidate in self.current_readonly_metal_mesh_payload_parameters:
                    return candidate
            for candidate in self.current_readonly_metal_mesh_payload_parameters:
                if display_name.startswith(
                    (f"{candidate}.", f"{candidate}->", f"{candidate}[")
                ):
                    return candidate
        root_name = self.assignment_target_root_name(target)
        if root_name in self.current_readonly_metal_mesh_payload_parameters:
            return root_name
        return None

    def record_readonly_metal_mesh_payload_assignment_alias(self, target, value):
        if self.argument_address_space(target) != "object_data":
            return None
        readonly_key = self.readonly_metal_mesh_payload_key(value)
        if readonly_key is None:
            return None
        target_name = self.assignment_target_display_name(target)
        if not target_name:
            return None
        reason = self.current_readonly_metal_mesh_payload_reasons.get(
            readonly_key, "const object_data"
        )
        self.current_readonly_metal_mesh_payload_parameters.add(target_name)
        self.current_readonly_metal_mesh_payload_reasons[target_name] = reason
        return target_name, reason

    def readonly_metal_mesh_payload_alias_assignment_diagnostic(self, target, value):
        alias = self.record_readonly_metal_mesh_payload_assignment_alias(target, value)
        if alias is None:
            return None
        target_name, reason = alias
        reason_text = self.readonly_metal_mesh_payload_reason_text(reason)
        return (
            "/* unsupported Metal mesh payload alias: mesh payload "
            f"'{target_name}' cannot store {reason_text} */"
        )

    def readonly_raw_buffer_assignment_diagnostic(self, target):
        root_name = self.assignment_target_root_name(target)
        if root_name not in self.current_readonly_raw_buffer_parameters:
            return None
        return (
            "/* unsupported Metal raw buffer store: readonly buffer "
            f"'{root_name}' cannot be written */"
        )

    def readonly_metal_parameter_assignment_diagnostic(self, target):
        root_name = self.assignment_target_root_name(target)
        if root_name not in self.current_readonly_metal_parameters:
            return None
        reason = self.current_readonly_metal_parameter_reasons.get(
            root_name, "readonly"
        )
        return (
            "/* unsupported Metal parameter store: parameter "
            f"'{root_name}' is {reason} */"
        )

    def readonly_metal_program_scope_global_assignment_diagnostic(self, target):
        root_name = self.assignment_target_root_name(target)
        if (
            root_name not in self.metal_program_scope_value_globals
            or root_name in self.local_variable_types
        ):
            return None
        target_name = self.assignment_target_display_name(target) or root_name
        if root_name in self.metal_program_scope_groupshared_globals:
            return (
                "/* unsupported Metal program-scope groupshared store: global "
                f"'{target_name}' cannot be written because Metal threadgroup "
                "storage must be declared inside a kernel */"
            )
        return (
            "/* unsupported Metal program-scope global store: global "
            f"'{target_name}' is emitted in the constant address space */"
        )

    def readonly_metal_mesh_payload_assignment_diagnostic(self, target):
        root_name = self.readonly_metal_mesh_payload_key(target)
        if root_name is None:
            return None
        reason = self.current_readonly_metal_mesh_payload_reasons.get(
            root_name, "const object_data"
        )
        reason_text = self.readonly_metal_mesh_payload_reason_text(reason)
        return (
            "/* unsupported Metal mesh payload store: mesh payload "
            f"'{root_name}' is {reason_text} */"
        )

    def readonly_metal_mesh_payload_reason_text(self, reason):
        if reason == "mesh stage payload":
            return "const object_data in mesh stages"
        if reason == "const-qualified payload parameter":
            return "const-qualified object_data"
        if reason == "address-space mismatch alias":
            return "read-only object_data after address-space mismatch"
        return reason

    def metal_mesh_payload_parameter_declaration(
        self, mapped_type, name, node=None, shader_type=None
    ):
        if not self.is_metal_mesh_payload_parameter(shader_type, node):
            return None

        readonly_reason = self.readonly_metal_mesh_payload_parameter_reason(
            shader_type, node
        )
        address_space = (
            "const object_data" if readonly_reason is not None else "object_data"
        )
        payload_type = self.metal_mesh_payload_value_type(mapped_type) or mapped_type
        return f"{address_space} {payload_type}& {name}"

    def readonly_metal_mesh_payload_parameter_reason(self, shader_type, node):
        if shader_type == "mesh":
            return "mesh stage payload"
        qualifiers = self.parameter_qualifier_names(node)
        if qualifiers & {"const", "constant"}:
            return "const-qualified payload parameter"
        return None

    def is_metal_mesh_payload_parameter(self, shader_type, node):
        if shader_type not in {"object", "task", "amplification", "mesh"}:
            return False
        return self.is_metal_mesh_payload_semantic(self.semantic_from_node(node))

    def is_metal_mesh_payload_semantic(self, semantic):
        if semantic is None:
            return False
        normalized = str(semantic).strip().lower().replace("-", "_")
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return normalized in {
            "payload",
            "mesh_payload",
            "hlsl_mesh_payload",
            "task_payload",
            "taskpayloadsharedext",
        }

    def is_explicit_metal_mesh_payload_semantic(self, semantic):
        if semantic is None:
            return False
        normalized = str(semantic).strip().lower().replace("-", "_")
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return normalized in {
            "mesh_payload",
            "hlsl_mesh_payload",
            "task_payload",
            "taskpayloadsharedext",
        }

    def metal_ray_payload_parameter_declaration(
        self, mapped_type, name, node=None, shader_type=None
    ):
        address_space = self.metal_ray_payload_address_space(shader_type)
        if address_space is None:
            return None
        ray_semantic_role = self.metal_ray_semantic_role(node, shader_type)
        if ray_semantic_role != "payload" and not (
            shader_type in {"ray_callable", "callable"}
            and ray_semantic_role == "callable_data"
        ):
            return None

        return f"{address_space} {mapped_type}& {name}"

    def metal_ray_payload_address_space(self, shader_type):
        if shader_type == "ray_generation":
            return "device"
        if shader_type in {
            "ray_intersection",
            "ray_any_hit",
            "ray_closest_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        }:
            return "ray_data"
        return None

    def structured_buffer_parameter_array_size(self, vtype, node=None):
        param_name = getattr(node, "name", None)
        function_hints = self.function_resource_array_size_hints.get(
            self.current_function_name, {}
        )
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return (
                self.safe_expression_to_string(vtype.size)
                if vtype.size is not None
                else function_hints.get(param_name, "")
            )

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        _, array_size = parse_array_type(type_string)
        return function_hints.get(param_name, "") if array_size is None else array_size

    def glsl_buffer_block_parameter_array_size(self, vtype, node=None):
        param_name = getattr(node, "name", None)
        function_hints = self.function_resource_array_size_hints.get(
            self.current_function_name, {}
        )
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return (
                self.safe_expression_to_string(vtype.size)
                if vtype.size is not None
                else function_hints.get(param_name, "")
            )

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        _, array_size = parse_array_type(type_string)
        return function_hints.get(param_name, "") if array_size is None else array_size

    def format_resource_parameter(self, resource_type, name, array_size):
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{resource_type}, {array_size}> {name}"
        return f"{resource_type} {name}"

    def resource_array_parameter(self, vtype, node=None):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_parameter_type(base_type):
                return None
            array_size = (
                self.safe_expression_to_string(vtype.size)
                if vtype.size is not None
                else self.function_resource_array_size_hints.get(
                    self.current_function_name, {}
                ).get(node.name, "")
            )
            return self.map_resource_type_with_format(base_type, node), array_size

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, array_size = parse_array_type(type_string)
        if not self.is_resource_parameter_type(base_type):
            return None
        return self.map_resource_type_with_format(base_type, node), (
            self.function_resource_array_size_hints.get(
                self.current_function_name, {}
            ).get(node.name, "")
            if array_size is None
            else array_size
        )

    def collect_resource_array_size_hints(self, ast):
        global_arrays = self.collect_unsized_resource_globals(ast)
        function_arrays = self.collect_unsized_resource_parameters(ast)
        fixed_global_array_sizes = self.collect_fixed_resource_global_sizes(ast)
        fixed_function_array_sizes = self.collect_fixed_resource_parameter_sizes(ast)
        if not (
            global_arrays
            or function_arrays
            or fixed_global_array_sizes
            or fixed_function_array_sizes
        ):
            return {}, {}

        return collect_resource_array_size_hints(
            global_arrays=global_arrays,
            function_arrays=function_arrays,
            fixed_global_array_sizes=fixed_global_array_sizes,
            fixed_function_array_sizes=fixed_function_array_sizes,
            functions=self.all_functions(ast),
            walk_nodes=self.iter_ast_nodes,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=1,
            format_size=str,
            initial_literal_int_constants=self.initial_literal_int_constants,
        )

    def collect_unsized_resource_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_resource_array_type(vtype):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_fixed_resource_global_sizes(self, ast):
        global_arrays = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            size = self.fixed_resource_array_size(vtype)
            if name and size is not None:
                global_arrays[name] = size
        return global_arrays

    def collect_unsized_resource_parameters(self, ast):
        function_arrays = {}
        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_unsized_resource_array_type(vtype):
                    function_arrays.setdefault(func_name, {})[param.name] = vtype
        return function_arrays

    def collect_fixed_resource_parameter_sizes(self, ast):
        function_arrays = {}
        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                size = self.fixed_resource_array_size(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                if size is not None:
                    function_arrays.setdefault(func_name, {})[param.name] = size
        return function_arrays

    def fixed_resource_array_size(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is None:
                return None
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_array_hint_type(base_type):
                return None
            size = self.literal_int_value(vtype.size, self.literal_int_constants)
            return size if size is not None and size > 0 else None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, size = parse_array_type(type_string)
        if size is None or not self.is_resource_array_hint_type(base_type):
            return None
        return max(size, 1)

    def is_unsized_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_resource_array_hint_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_resource_array_hint_type(base_type)

    def is_resource_array_hint_type(self, vtype):
        return (
            self.is_resource_parameter_type(vtype)
            or str(self.resource_base_type(vtype))
            in self.glsl_buffer_block_struct_names
        )

    def unsupported_stage_types(self):
        return set()

    def validate_supported_stage_types(self, ast, target_stage=None):
        unsupported_stages = []
        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name in self.unsupported_stage_types() and stage_matches(
                target_stage, stage_name
            ):
                unsupported_stages.append(stage_name)

        if unsupported_stages:
            stage_list = ", ".join(sorted(unsupported_stages))
            raise ValueError(
                f"Metal output does not support stage type(s): {stage_list}"
            )

    def has_geometry_stage(self, ast, target_stage=None):
        if not stage_matches(target_stage, "geometry"):
            return False

        for stage_type in getattr(ast, "stages", {}) or {}:
            if normalize_stage_name(stage_type) == "geometry":
                return True

        for func in getattr(ast, "functions", []) or []:
            qualifiers = getattr(func, "qualifiers", None)
            if qualifiers is None:
                qualifiers = [getattr(func, "qualifier", None)]
            if any(
                normalize_stage_name(qualifier) == "geometry"
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_metal_geometry_stream_helpers(self):
        return (
            "template <typename T>\n"
            "struct CrossGLMetalPointStream {\n"
            "    void Append(T value) { }\n"
            "    void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CrossGLMetalLineStream {\n"
            "    void Append(T value) { }\n"
            "    void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CrossGLMetalTriangleStream {\n"
            "    void Append(T value) { }\n"
            "    void RestartStrip() { }\n"
            "};\n\n"
        )

    def has_tessellation_stage(self, ast, target_stage=None):
        tessellation_stages = {"tessellation_control", "tessellation_evaluation"}
        if not any(stage_matches(target_stage, stage) for stage in tessellation_stages):
            return False

        for stage_type in getattr(ast, "stages", {}) or {}:
            if normalize_stage_name(stage_type) in tessellation_stages:
                return True

        for func in getattr(ast, "functions", []) or []:
            qualifiers = getattr(func, "qualifiers", None)
            if qualifiers is None:
                qualifiers = [getattr(func, "qualifier", None)]
            if any(
                normalize_stage_name(qualifier) in tessellation_stages
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_metal_tessellation_patch_helpers(self):
        return (
            "template <typename T, int N>\n"
            "struct CrossGLMetalInputPatch {\n"
            "    T control_points[N];\n"
            "    T operator[](uint index) const { return control_points[index]; }\n"
            "};\n\n"
            "template <typename T, int N>\n"
            "struct CrossGLMetalOutputPatch {\n"
            "    T control_points[N];\n"
            "    T operator[](uint index) const { return control_points[index]; }\n"
            "};\n\n"
        )

    def stage_entry_types(self):
        return {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        }

    def stage_entry_base_name(self, stage_name, func):
        func_name = getattr(func, "name", None) or "main"
        if self.function_has_explicit_stage_entry_attribute(func):
            return func_name
        if stage_name == "vertex":
            return f"vertex_{func_name}"
        if stage_name == "fragment":
            return f"fragment_{func_name}"
        if stage_name in {"compute", "ray_generation"}:
            return f"kernel_{func_name}"
        if stage_name == "geometry":
            return f"geometry_{func_name}"
        if stage_name == "tessellation_control":
            return f"tessellation_control_{func_name}"
        if stage_name == "tessellation_evaluation":
            return f"tessellation_evaluation_{func_name}"
        if stage_name in {"mesh", "object", "task", "amplification"}:
            stage_keyword = "mesh" if stage_name == "mesh" else "object"
            return f"{stage_keyword}_{func_name}"

        rt_stage_map = {
            "ray_intersection": "intersection",
            "ray_any_hit": "anyhit",
            "ray_closest_hit": "closesthit",
            "ray_miss": "miss",
            "ray_callable": "callable",
            "intersection": "intersection",
            "anyhit": "anyhit",
            "closesthit": "closesthit",
            "miss": "miss",
            "callable": "callable",
        }
        stage_keyword = rt_stage_map.get(stage_name)
        if stage_keyword:
            return f"{stage_keyword}_{func_name}"
        return func_name

    def function_has_explicit_stage_entry_attribute(self, func):
        if getattr(func, "preserve_stage_entry_name", False):
            return True
        return any(
            str(getattr(attr, "name", "")).lower() == "stage_entry"
            for attr in getattr(func, "attributes", []) or []
        )

    def function_stage_qualifier(self, func):
        qualifiers = getattr(func, "qualifiers", None)
        if qualifiers:
            return qualifiers[0]

        qualifier = getattr(func, "qualifier", None)
        if qualifier:
            return qualifier

        stage_entry_types = self.stage_entry_types()
        for attr in getattr(func, "attributes", []) or []:
            stage_name = normalize_stage_name(getattr(attr, "name", None))
            if stage_name in stage_entry_types:
                return stage_name
        return None

    def stage_entry_names(self, ast, target_stage=None):
        stage_entry_types = self.stage_entry_types()
        entries = collect_stage_entry_records(ast, target_stage, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, target_stage, stage_entry_types
        )
        return assign_stage_entry_names(entries, used_names, self.stage_entry_base_name)

    def metal_stage_entry_functions(self, ast, expected_stage_name, target_stage=None):
        if not stage_matches(target_stage, expected_stage_name):
            return []

        functions = []
        for func in getattr(ast, "functions", []) or []:
            if function_stage_name(func) == expected_stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            if normalize_stage_name(stage_type) != expected_stage_name:
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
        return functions

    def collect_metal_stage_entry_parameter_struct_names(
        self, ast, target_stage, expected_stage_name
    ):
        struct_names = set()
        for func in self.metal_stage_entry_functions(
            ast, expected_stage_name, target_stage
        ):
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                struct_name = self.metal_parameter_user_struct_type(parameter)
                if struct_name is not None:
                    struct_names.add(struct_name)
        return struct_names

    def collect_metal_stage_entry_return_struct_names(
        self, ast, target_stage, expected_stage_name
    ):
        struct_names = set()
        for func in self.metal_stage_entry_functions(
            ast, expected_stage_name, target_stage
        ):
            struct_node = self.metal_return_struct(func)
            if struct_node is not None:
                struct_names.add(struct_node.name)
        return struct_names

    def metal_parameter_user_struct_type(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None
        mapped_type = self.map_type(type_name)
        base_type, _array_suffix = split_array_type_suffix(str(mapped_type))
        struct_name = base_type.split("<", 1)[0].strip()
        if struct_name not in self.structs_by_name:
            return None
        return struct_name

    def metal_return_struct(self, func):
        return_type = self.type_name_string(
            getattr(func, "return_type", getattr(func, "vtype", None))
        )
        if not return_type:
            return None
        base_type = return_type.split("<", 1)[0].split("[", 1)[0].strip()
        return self.structs_by_name.get(base_type)

    def metal_default_struct_member_semantics(self, struct_node):
        struct_name = getattr(struct_node, "name", None)
        defaults = {}
        if struct_name in self.metal_vertex_entry_input_struct_names:
            defaults.update(
                self.metal_default_vertex_input_member_semantics(struct_node)
            )
        elif struct_name in self.metal_vertex_entry_output_struct_names:
            defaults.update(
                self.metal_default_vertex_output_member_semantics(struct_node)
            )
        elif struct_name in self.metal_fragment_entry_output_struct_names:
            defaults.update(
                self.metal_default_fragment_output_member_semantics(struct_node)
            )
        if struct_name in self.metal_stage_io_struct_names:
            defaults.update(
                self.metal_default_user_stage_io_member_semantics(struct_node, defaults)
            )
        return defaults

    def metal_default_vertex_input_member_semantics(self, struct_node):
        used_attributes = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.metal_vertex_input_location_semantic(member)
            if semantic is None:
                semantic = self.semantic_from_node(member)
            attribute_index = self.metal_attribute_index_from_semantic(semantic)
            if attribute_index is not None:
                span = self.metal_stage_io_member_attribute_span(member)
                used_attributes.update(range(attribute_index, attribute_index + span))

        defaults = {}
        next_attribute = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            location_semantic = self.metal_vertex_input_location_semantic(member)
            if location_semantic is not None:
                defaults[member_name] = location_semantic
                continue
            if self.semantic_from_node(member) is not None:
                continue
            if not self.metal_can_default_stage_io_semantic(
                self.struct_member_raw_type(member)
            ):
                continue

            while next_attribute in used_attributes:
                next_attribute += 1
            defaults[member_name] = f"attribute({next_attribute})"
            span = self.metal_stage_io_member_attribute_span(member)
            used_attributes.update(range(next_attribute, next_attribute + span))
            next_attribute += span
        return defaults

    def explicit_location_attribute_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "location":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            location = self.binding_index_value(arguments[0])
            if location is not None:
                return location
        return None

    def metal_vertex_input_location_semantic(self, node):
        location = self.explicit_location_attribute_index(node)
        if location is None:
            return None
        return f"attribute({location})"

    def metal_vertex_input_location_attribute(self, node):
        semantic = self.metal_vertex_input_location_semantic(node)
        if semantic is None:
            return None
        return self.map_semantic(semantic)

    def metal_default_user_stage_io_member_semantics(
        self, struct_node, existing_defaults=None
    ):
        defaults = {}
        existing_defaults = existing_defaults or {}
        used_attributes = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = existing_defaults.get(getattr(member, "name", None))
            if semantic is None:
                semantic = self.semantic_from_node(member)
            attribute_index = self.metal_attribute_index_from_semantic(semantic)
            if attribute_index is not None:
                span = self.metal_stage_io_member_attribute_span(member)
                used_attributes.update(range(attribute_index, attribute_index + span))

        next_attribute = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in existing_defaults:
                continue
            if not self.is_metal_user_stage_io_semantic(
                self.semantic_from_node(member)
            ):
                continue
            if not self.metal_can_default_stage_io_semantic(
                self.struct_member_raw_type(member)
            ):
                continue

            while next_attribute in used_attributes:
                next_attribute += 1
            defaults[member_name] = f"attribute({next_attribute})"
            span = self.metal_stage_io_member_attribute_span(member)
            used_attributes.update(range(next_attribute, next_attribute + span))
            next_attribute += span
        return defaults

    def metal_struct_member_effective_semantic(
        self, member, default_member_semantics, struct_name=None
    ):
        semantic = self.semantic_from_node(member)
        default_semantic = (default_member_semantics or {}).get(
            getattr(member, "name", None)
        )
        if semantic is None:
            return default_semantic
        if (
            struct_name in getattr(self, "metal_stage_io_struct_names", set())
            and default_semantic is not None
            and self.is_metal_user_stage_io_semantic(semantic)
        ):
            return default_semantic
        return semantic

    def is_metal_user_stage_io_semantic(self, semantic):
        if semantic is None:
            return False
        if self.metal_attribute_index_from_semantic(semantic) is not None:
            return False
        canonical = self.canonical_metal_semantic(semantic)
        if canonical is None:
            return False
        canonical = str(canonical)
        if canonical.startswith("attribute("):
            return False
        if canonical.startswith("color(") or canonical.startswith("depth("):
            return False
        return canonical not in {
            "barycentric_coord",
            "clip_distance",
            "instance_id",
            "is_front_facing",
            "payload",
            "point_coord",
            "point_size",
            "position",
            "primitive_id",
            "sample_id",
            "sample_mask",
            "stencil_ref",
            "thread_index_in_threadgroup",
            "thread_position_in_grid",
            "thread_position_in_threadgroup",
            "threadgroup_position_in_grid",
            "threads_per_threadgroup",
            "vertex_id",
        }

    def metal_stage_io_member_attribute_span(self, member):
        array_info = self.metal_stage_io_member_array_info(member)
        if array_info is not None:
            _element_type, size = array_info
            return size

        mapped_type = self.map_type(self.struct_member_raw_type(member))
        dimensions = self.metal_matrix_dimensions(mapped_type)
        if dimensions is not None:
            _component_prefix, columns, _rows = dimensions
            return columns
        return 1

    def metal_default_vertex_output_member_semantics(self, struct_node):
        defaults = {}
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            normalized_name = str(member_name).replace("_", "").lower()
            if normalized_name not in {
                "position",
                "clipposition",
                "glposition",
            }:
                continue
            if self.semantic_from_node(member) is not None:
                continue
            if self.map_type(self.struct_member_raw_type(member)) != "float4":
                continue
            defaults[member_name] = "position"
        return defaults

    def metal_vertex_output_position_fallback_member_name(self, struct_node):
        struct_name = getattr(struct_node, "name", None)
        if struct_name not in self.metal_vertex_entry_output_struct_names:
            return None
        if self.metal_vertex_output_has_position_semantic(struct_node):
            return None

        existing_names = {
            getattr(member, "name", None)
            for member in getattr(struct_node, "members", []) or []
        }
        return self.unique_metal_generated_name("__crossgl_position", existing_names)

    def metal_vertex_output_has_position_semantic(self, struct_node):
        default_semantics = self.metal_default_vertex_output_member_semantics(
            struct_node
        )
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                semantic = default_semantics.get(getattr(member, "name", None))
            if self.canonical_metal_semantic(semantic) == "position":
                return True
        return False

    def metal_vertex_output_position_fallback_member_for_type(self, return_type):
        base_type = (
            self.type_name_string(return_type).split("<", 1)[0].split("[", 1)[0].strip()
        )
        struct_node = self.structs_by_name.get(base_type)
        if struct_node is None:
            return None
        return self.metal_vertex_output_position_fallback_member_name(struct_node)

    def generate_metal_vertex_output_position_fallback_return(
        self, return_expr, indent_str
    ):
        fallback_member = self.metal_vertex_output_position_fallback_member_for_type(
            self.current_function_return_type
        )
        if fallback_member is None:
            return None

        return_type = self.map_type(self.current_function_return_type)
        reserved_names = set(self.local_variable_types)
        result_name = self.unique_metal_generated_name("_crossglReturn", reserved_names)
        return (
            f"{indent_str}{return_type} {result_name} = {return_expr};\n"
            f"{indent_str}{result_name}.{fallback_member} = "
            "float4(0.0, 0.0, 0.0, 1.0);\n"
            f"{indent_str}return {result_name};\n"
        )

    def metal_default_fragment_output_member_semantics(self, struct_node):
        defaults = {}
        used_color_indices = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            color_index = self.metal_color_index_from_semantic(semantic)
            if color_index is not None:
                used_color_indices.add(color_index)

        next_color_index = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or self.semantic_from_node(member) is not None:
                continue
            mapped_type = self.map_type(self.struct_member_raw_type(member))
            normalized_name = str(member_name).replace("_", "").lower()
            if normalized_name in {"depth", "fragdepth", "glfragdepth"}:
                if mapped_type in {"float", "half"}:
                    defaults[member_name] = "depth(any)"
                continue
            if mapped_type != "float4":
                continue
            while next_color_index in used_color_indices:
                next_color_index += 1
            defaults[member_name] = f"color({next_color_index})"
            used_color_indices.add(next_color_index)
            next_color_index += 1
        return defaults

    def metal_can_default_stage_io_semantic(self, member_type):
        type_name = self.type_name_string(member_type)
        if not type_name:
            return False

        base_type, _array_size = parse_array_type(str(type_name))
        mapped_type = self.map_type(base_type)
        if mapped_type in self.structs_by_name:
            return False
        return not self.is_resource_parameter_type(mapped_type)

    def metal_color_index_from_semantic(self, semantic):
        if semantic is None:
            return None
        mapped_semantic = self.semantic_map.get(semantic, semantic)
        mapped_semantic = str(mapped_semantic)
        if not mapped_semantic.startswith("color("):
            return None
        if not mapped_semantic.endswith(")"):
            return None
        index = mapped_semantic[len("color(") : -1]
        return int(index) if index.isdigit() else None

    def metal_attribute_index_from_semantic(self, semantic):
        if semantic is None:
            return None
        mapped_semantic = self.semantic_map.get(semantic, semantic)
        mapped_semantic = str(mapped_semantic)
        if not mapped_semantic.startswith("attribute("):
            return None
        if not mapped_semantic.endswith(")"):
            return None
        index = mapped_semantic[len("attribute(") : -1]
        return int(index) if index.isdigit() else None

    def metal_attribute_index_from_attribute(self, attribute):
        if not attribute:
            return None
        match = re.search(r"\[\[\s*attribute\((\d+)\)\s*\]\]", str(attribute))
        return int(match.group(1)) if match else None

    def uses_metal_raytracing_namespace(self, ast, global_vars, functions):
        for stage_type in getattr(ast, "stages", {}) or {}:
            if normalize_stage_name(stage_type) in {
                "ray_generation",
                "ray_intersection",
                "ray_any_hit",
                "ray_closest_hit",
                "ray_miss",
                "ray_callable",
                "intersection",
                "anyhit",
                "closesthit",
                "miss",
                "callable",
            }:
                return True

        for variable in global_vars:
            vtype, _ = self.global_resource_shape(variable)
            if (
                self.is_acceleration_structure_type(vtype)
                or self.is_visible_function_table_type(vtype)
                or self.is_intersection_function_table_type(vtype)
            ):
                return True

        for func in functions:
            for node in self.iter_ast_nodes(getattr(func, "body", [])):
                if isinstance(node, (RayTracingOpNode, RayQueryOpNode)):
                    return True
        return False

    def metal_mesh_stage_attribute(self, shader_type, func, execution_config=None):
        stage_keyword = "mesh" if shader_type == "mesh" else "object"
        attributes = [stage_keyword]
        threadgroup_limit = self.metal_stage_threadgroup_limit(
            func, shader_type, execution_config
        )
        if threadgroup_limit:
            attributes.append(f"max_total_threads_per_threadgroup({threadgroup_limit})")
        return f"[[{', '.join(attributes)}]]"

    def metal_ray_intersection_stage_attribute(self, func, raw_return_type):
        primitive_type = self.metal_ray_intersection_primitive_type(
            func, raw_return_type
        )
        return f"[[intersection({primitive_type})]]"

    def metal_ray_intersection_primitive_type(self, func, raw_return_type):
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            arguments = getattr(attr, "arguments", []) or []
            if normalized in {"bounding_box", "boundingbox"}:
                return "bounding_box"
            if normalized in {"triangle", "triangles"}:
                return "triangle"
            if normalized in {"intersection", "primitive_type"} and arguments:
                requested_value = self.attribute_value_to_string(arguments[0])
                if requested_value is None:
                    continue
                requested = str(requested_value).strip().lower().replace("-", "_")
                if requested in {"bounding_box", "boundingbox"}:
                    return "bounding_box"
                if requested in {"triangle", "triangles"}:
                    return "triangle"

        return "triangle" if raw_return_type == "bool" else "bounding_box"

    def metal_stage_threadgroup_limit(self, func, shader_type, execution_config=None):
        if shader_type not in {"mesh", "object", "task", "amplification"}:
            return None

        arguments = self.metal_stage_attribute_arguments(
            func, "max_total_threads_per_threadgroup"
        )
        if arguments:
            if len(arguments) != 1:
                raise ValueError(
                    f"Metal {shader_type} stage max_total_threads_per_threadgroup "
                    "requires exactly one argument"
                )
            return self.attribute_value_to_string(arguments[0])

        if execution_config:
            return self.metal_local_size_threadgroup_total(execution_config)
        return None

    def metal_stage_attribute_arguments(self, func, attribute_name):
        for attr in getattr(func, "attributes", []) or []:
            if self.metal_stage_control_attribute_name(attr) == attribute_name:
                return getattr(attr, "arguments", []) or []
        return []

    def metal_geometry_maxvertexcount(self, func, stage_node=None):
        arguments = self.metal_stage_attribute_arguments(func, "maxvertexcount")
        if arguments:
            if len(arguments) != 1:
                raise ValueError(
                    "Metal geometry stage maxvertexcount requires exactly one argument"
                )
            value_text = self.attribute_value_to_string(arguments[0])
        else:
            value_text = None
            if stage_node is not None:
                value_text = stage_layout_entry_value(
                    stage_node, "max_vertices", "out"
                ) or stage_layout_entry_value(
                    stage_node,
                    "maxvertexcount",
                    "out",
                )
            if value_text is None:
                raise ValueError(
                    "Metal geometry stage requires maxvertexcount attribute"
                )
        value = self.literal_int_value(value_text, self.literal_int_constants)
        if value is not None and value <= 0:
            raise ValueError(
                f"Metal geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def metal_parameter_qualifiers(self, parameter):
        qualifiers = []
        allowed_qualifiers = {
            "const",
            "in",
            "out",
            "inout",
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        for attr in getattr(parameter, "attributes", []) or []:
            if getattr(attr, "name", None) != "primitive":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            primitive = self.attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def metal_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.metal_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def metal_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def metal_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            return self.literal_int_value(size, self.literal_int_constants)

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, array_size = parse_array_type(type_name)
        if array_size is None:
            return None
        return self.literal_int_value(array_size, self.literal_int_constants)

    def validate_metal_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.metal_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.metal_parameter_is_array(parameter):
                raise ValueError(
                    "Metal geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.metal_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise ValueError(
                    "Metal geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_metal_geometry_stage(self, func, parameters, stage_node=None):
        self.metal_geometry_maxvertexcount(func, stage_node=stage_node)

        if not any(
            self.metal_geometry_stream_info(self.parameter_raw_type(param))
            for param in parameters
        ):
            raise ValueError(
                "Metal geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.metal_geometry_input_primitive_qualifier(param) for param in parameters
        ):
            raise ValueError(
                "Metal geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_metal_geometry_input_primitive_arity(parameters)

    def generate_metal_geometry_stage_comments(
        self,
        func,
        parameters,
        stage_node=None,
    ):
        maxvertexcount = self.metal_geometry_maxvertexcount(
            func,
            stage_node=stage_node,
        )
        input_descriptions = []
        stream_descriptions = []

        for parameter in parameters:
            primitive = self.metal_geometry_input_primitive_qualifier(parameter)
            if primitive is not None:
                input_descriptions.append(f"{primitive}:{parameter.name}")

            stream_info = self.metal_geometry_stream_info(
                self.parameter_raw_type(parameter)
            )
            if stream_info is not None:
                stream_name, output_type = stream_info
                stream_descriptions.append(
                    f"{stream_name}:{parameter.name}->{output_type}"
                )

        lines = [f"// CrossGL geometry stage: maxvertexcount={maxvertexcount}\n"]
        if input_descriptions:
            lines.append(
                "// CrossGL geometry input primitive: "
                f"{', '.join(input_descriptions)}\n"
            )
        if stream_descriptions:
            lines.append(
                "// CrossGL geometry output stream: "
                f"{', '.join(stream_descriptions)}\n"
            )
        return "".join(lines)

    def metal_tessellation_stage_attribute_value(self, func, attribute_name):
        arguments = self.metal_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"Metal tessellation stage attribute {attribute_name} requires "
                "exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def metal_tessellation_output_control_points(self, func):
        output_control_points = self.metal_tessellation_stage_attribute_value(
            func, "outputcontrolpoints"
        )
        if output_control_points is None:
            output_control_points = self.metal_tessellation_stage_attribute_value(
                func, "vertices"
            )
        return output_control_points

    def metal_tessellation_positive_int_attribute(
        self, func, attribute_name, stage_name, display_name=None
    ):
        value_text = self.metal_tessellation_stage_attribute_value(func, attribute_name)
        if value_text is None:
            return None
        value = self.literal_int_value(value_text, self.literal_int_constants)
        display_name = display_name or attribute_name
        if value is None:
            raise ValueError(
                f"Metal {stage_name} stage {display_name} must be an integer "
                f"constant, got {value_text}"
            )
        if value <= 0:
            raise ValueError(
                f"Metal {stage_name} stage {display_name} ({value}) must be " "positive"
            )
        return value

    def metal_tessellation_domain(self, func):
        value = self.metal_tessellation_stage_attribute_value(func, "domain")
        if value is None:
            return None
        domain_name = str(value).strip().strip('"').lower()
        domain_map = {
            "tri": "triangle",
            "triangle": "triangle",
            "triangles": "triangle",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }
        mapped = domain_map.get(domain_name)
        if mapped is None:
            raise ValueError(
                f"Metal tessellation domain cannot be represented: {value}"
            )
        return mapped

    def metal_tessellation_partitioning(self, func):
        value = self.metal_tessellation_stage_attribute_value(func, "partitioning")
        if value is None:
            return None
        partition_name = str(value).strip().strip('"').lower()
        partition_map = {
            "integer": "integer",
            "equal": "integer",
            "equal_spacing": "integer",
            "fractional_even": "fractional_even",
            "fractional_even_spacing": "fractional_even",
            "fractional_odd": "fractional_odd",
            "fractional_odd_spacing": "fractional_odd",
        }
        mapped = partition_map.get(partition_name)
        if mapped is None:
            raise ValueError(
                f"Metal tessellation partitioning cannot be represented: {value}"
            )
        return mapped

    def metal_tessellation_output_topology(self, func, domain):
        value = self.metal_tessellation_stage_attribute_value(func, "outputtopology")
        if value is None:
            return None

        topology_name = str(value).strip().strip('"').lower()
        if topology_name in {"triangle_cw", "triangles_cw"}:
            if domain != "triangle":
                raise ValueError(
                    "Metal tessellation outputtopology triangle_cw requires "
                    "triangle domain"
                )
            return "triangle_cw"
        if topology_name in {"triangle_ccw", "triangles_ccw"}:
            if domain != "triangle":
                raise ValueError(
                    "Metal tessellation outputtopology triangle_ccw requires "
                    "triangle domain"
                )
            return "triangle_ccw"
        if topology_name in {"line", "lines"}:
            if domain != "isoline":
                raise ValueError(
                    "Metal tessellation outputtopology line requires isoline domain"
                )
            return "line"
        if topology_name in {"point", "points"}:
            return "point"

        raise ValueError(
            f"Metal tessellation outputtopology cannot be represented: {value}"
        )

    def metal_tessellation_patch_constant_function_name(self, func):
        return self.metal_tessellation_stage_attribute_value(func, "patchconstantfunc")

    def metal_tessellation_patch_parameters(self, parameters, patch_type):
        for parameter in parameters:
            info = self.metal_tessellation_patch_info(
                self.parameter_raw_type(parameter)
            )
            if info is not None and info["kind"] == patch_type:
                yield parameter, info

    def metal_tessellation_patch_control_point_count(self, parameters, patch_type):
        for _parameter, info in self.metal_tessellation_patch_parameters(
            parameters, patch_type
        ):
            return info["count_value"]
        return None

    def validate_metal_tessellation_patch_parameter_signature(
        self, parameter, info, stage_name
    ):
        patch_type = info["kind"]
        parameter_name = getattr(parameter, "name", None)
        name_clause = (
            f" parameter '{parameter_name}'" if parameter_name else " parameter"
        )
        type_name = self.type_name_string(self.parameter_raw_type(parameter))

        if not info["valid"]:
            raise ValueError(
                f"Metal {stage_name} stage {patch_type}{name_clause} must use "
                f"{patch_type}<T, N>; found '{type_name or patch_type}'"
            )

        if info["element_type"] is None:
            raise ValueError(
                f"Metal {stage_name} stage {patch_type}{name_clause} must include "
                "an element type"
            )

        control_points = info["count_value"]
        if control_points is None:
            raise ValueError(
                f"Metal {stage_name} stage {patch_type}{name_clause} control "
                f"point count '{info['count_text']}' must be an integer constant"
            )
        if control_points <= 0:
            raise ValueError(
                f"Metal {stage_name} stage {patch_type}{name_clause} control "
                f"point count ({control_points}) must be positive"
            )
        if control_points > 32:
            raise ValueError(
                f"Metal {stage_name} stage {patch_type}{name_clause} control "
                f"point count ({control_points}) must be at most 32"
            )

    def validate_metal_tessellation_patch_parameter_signatures(
        self, parameters, patch_type, stage_name
    ):
        for parameter, info in self.metal_tessellation_patch_parameters(
            parameters, patch_type
        ):
            self.validate_metal_tessellation_patch_parameter_signature(
                parameter, info, stage_name
            )

    def validate_metal_tessellation_control_stage(self, func, parameters):
        stage_name = "tessellation_control"
        domain = self.metal_tessellation_domain(func)
        if domain is None:
            raise ValueError(
                "Metal tessellation_control stage requires domain attribute"
            )
        if self.metal_tessellation_partitioning(func) is None:
            raise ValueError(
                "Metal tessellation_control stage requires partitioning attribute"
            )
        if self.metal_tessellation_output_topology(func, domain) is None:
            raise ValueError(
                "Metal tessellation_control stage requires outputtopology attribute"
            )

        input_patch_parameters = list(
            self.metal_tessellation_patch_parameters(parameters, "InputPatch")
        )
        if not input_patch_parameters:
            raise ValueError(
                "Metal tessellation_control stage parameters must include an "
                "InputPatch<T, N> parameter"
            )
        self.validate_metal_tessellation_patch_parameter_signatures(
            parameters, "InputPatch", stage_name
        )

        output_control_points = self.metal_tessellation_positive_int_attribute(
            func, "outputcontrolpoints", stage_name
        )
        if output_control_points is None:
            output_control_points = self.metal_tessellation_positive_int_attribute(
                func, "vertices", stage_name, display_name="outputcontrolpoints"
            )
        if output_control_points is None:
            raise ValueError(
                "Metal tessellation_control stage requires outputcontrolpoints "
                "attribute"
            )
        if output_control_points > 32:
            raise ValueError(
                "Metal tessellation_control stage outputcontrolpoints "
                f"({output_control_points}) must be at most 32"
            )

        input_control_points = self.metal_tessellation_patch_control_point_count(
            parameters, "InputPatch"
        )
        if (
            input_control_points is not None
            and output_control_points != input_control_points
        ):
            raise ValueError(
                "Metal tessellation_control stage outputcontrolpoints "
                f"({output_control_points}) must match the InputPatch control "
                f"point count ({input_control_points})"
            )

        patch_constant_func = self.metal_tessellation_patch_constant_function_name(func)
        if patch_constant_func is None:
            raise ValueError(
                "Metal tessellation_control stage requires patchconstantfunc "
                "attribute"
            )
        if patch_constant_func not in self.user_function_names:
            raise ValueError(
                "Metal tessellation_control stage patchconstantfunc "
                f"'{patch_constant_func}' must name a function in the program"
            )

        if not any(
            self.metal_parameter_has_semantic(parameter, "SV_OutputControlPointID")
            for parameter in parameters
        ):
            raise ValueError(
                "Metal tessellation_control stage parameters must include an "
                "SV_OutputControlPointID parameter"
            )
        self.validate_metal_scalar_integer_semantic_types(
            parameters,
            stage_name,
            ("SV_OutputControlPointID", "SV_PrimitiveID"),
        )
        self.validate_metal_tessellation_max_factor(func, stage_name)

    def validate_metal_tessellation_evaluation_stage(self, func, parameters):
        stage_name = "tessellation_evaluation"
        domain = self.metal_tessellation_domain(func)
        if domain is None:
            raise ValueError(
                "Metal tessellation_evaluation stage requires domain attribute"
            )
        self.metal_tessellation_output_topology(func, domain)

        output_patch_parameters = list(
            self.metal_tessellation_patch_parameters(parameters, "OutputPatch")
        )
        if not output_patch_parameters:
            raise ValueError(
                "Metal tessellation_evaluation stage parameters must include an "
                "OutputPatch<T, N> parameter"
            )
        self.validate_metal_tessellation_patch_parameter_signatures(
            parameters, "OutputPatch", stage_name
        )

        if not any(
            self.metal_parameter_has_semantic(parameter, "SV_DomainLocation")
            for parameter in parameters
        ):
            raise ValueError(
                "Metal tessellation_evaluation stage parameters must include an "
                "SV_DomainLocation parameter"
            )
        self.validate_metal_domain_location_type(parameters)
        self.validate_metal_domain_location_components(func, parameters)
        self.validate_metal_scalar_integer_semantic_types(
            parameters,
            stage_name,
            ("SV_PrimitiveID",),
        )

    def validate_metal_tessellation_max_factor(self, func, stage_name):
        max_tess_factor_text = self.metal_tessellation_stage_attribute_value(
            func, "maxtessfactor"
        )
        if max_tess_factor_text is None:
            return
        try:
            max_tess_factor = float(str(max_tess_factor_text).strip().strip('"'))
        except ValueError as exc:
            raise ValueError(
                "Metal tessellation_control stage maxtessfactor "
                f"'{max_tess_factor_text}' must be a numeric literal"
            ) from exc
        if not (1.0 <= max_tess_factor <= 64.0):
            raise ValueError(
                "Metal tessellation_control stage maxtessfactor "
                f"({max_tess_factor_text}) must be in the range 1.0..64.0"
            )

    def validate_metal_tessellation_stage(self, func, parameters, shader_type):
        if shader_type == "tessellation_control":
            self.validate_metal_tessellation_control_stage(func, parameters)
        elif shader_type == "tessellation_evaluation":
            self.validate_metal_tessellation_evaluation_stage(func, parameters)

    def validate_metal_domain_location_type(self, parameters):
        for parameter in parameters:
            if not self.metal_parameter_has_semantic(parameter, "SV_DomainLocation"):
                continue

            base_type, array_suffix = self.metal_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue

            floating_scalar_bases = ("float", "half", "double")
            is_floating_type = False
            for scalar_base in floating_scalar_bases:
                if not base_type.startswith(scalar_base):
                    continue
                suffix = base_type[len(scalar_base) :]
                if suffix in {"", "2", "3", "4"}:
                    is_floating_type = True
                break

            if array_suffix or not is_floating_type:
                raise ValueError(
                    "Metal tessellation_evaluation stage SV_DomainLocation "
                    f"parameter '{parameter.name}' must be a floating-point "
                    "scalar or vector"
                )

    def validate_metal_domain_location_components(self, func, parameters):
        domain = self.metal_tessellation_domain(func)
        expected_counts = {
            "triangle": 3,
            "quad": 2,
            "isoline": 2,
        }
        expected_count = expected_counts.get(domain)
        if expected_count is None:
            return

        for parameter in parameters:
            if not self.metal_parameter_has_semantic(parameter, "SV_DomainLocation"):
                continue

            component_count = self.metal_parameter_component_count(parameter)
            if component_count is None:
                continue
            if component_count != expected_count:
                raise ValueError(
                    "Metal tessellation_evaluation stage SV_DomainLocation "
                    f"parameter '{parameter.name}' must have {expected_count} "
                    f"component(s) for {domain} domains, got {component_count}"
                )

    def validate_metal_scalar_integer_semantic_type(
        self, parameters, shader_type, semantic
    ):
        for parameter in parameters:
            if not self.metal_parameter_has_semantic(parameter, semantic):
                continue

            base_type, array_suffix = self.metal_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue
            if array_suffix or base_type not in {"int", "uint"}:
                raise ValueError(
                    f"Metal {shader_type} stage {semantic} parameter "
                    f"'{parameter.name}' must be scalar int or uint"
                )

    def validate_metal_scalar_integer_semantic_types(
        self, parameters, shader_type, semantics
    ):
        for semantic in semantics:
            self.validate_metal_scalar_integer_semantic_type(
                parameters, shader_type, semantic
            )

    def metal_parameter_has_semantic(self, parameter, expected_semantic):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return False
        return str(semantic).lower() == expected_semantic.lower()

    def metal_parameter_mapped_base_and_array_suffix(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        type_name = self.type_name_string(raw_type)
        if not type_name:
            return None, None
        mapped_type = self.map_type(type_name)
        return split_array_type_suffix(str(mapped_type))

    def metal_parameter_component_count(self, parameter):
        base_type, _array_suffix = self.metal_parameter_mapped_base_and_array_suffix(
            parameter
        )
        if base_type is None:
            return None

        scalar_bases = (
            "float",
            "half",
            "double",
            "int",
            "uint",
            "bool",
        )
        for scalar_base in scalar_bases:
            if not base_type.startswith(scalar_base):
                continue
            suffix = base_type[len(scalar_base) :]
            if suffix in {"2", "3", "4"}:
                return int(suffix)
            if suffix == "":
                return 1
        return None

    def generate_metal_tessellation_stage_comments(self, func, parameters, shader_type):
        metadata = []
        domain = self.metal_tessellation_domain(func)
        if domain:
            metadata.append(f"domain={domain}")
        partitioning = self.metal_tessellation_partitioning(func)
        if partitioning:
            metadata.append(f"partitioning={partitioning}")
        output_topology = self.metal_tessellation_output_topology(func, domain)
        if output_topology:
            metadata.append(f"outputtopology={output_topology}")
        output_control_points = self.metal_tessellation_output_control_points(func)
        if output_control_points:
            metadata.append(f"outputcontrolpoints={output_control_points}")
        patch_constant_func = self.metal_tessellation_patch_constant_function_name(func)
        if patch_constant_func:
            metadata.append(f"patchconstantfunc={patch_constant_func}")
        max_tess_factor = self.metal_tessellation_stage_attribute_value(
            func, "maxtessfactor"
        )
        if max_tess_factor:
            metadata.append(f"maxtessfactor={max_tess_factor}")

        lines = [
            f"// CrossGL {shader_type} stage"
            f"{': ' + ', '.join(metadata) if metadata else ''}\n"
        ]

        patch_descriptions = []
        for parameter in parameters:
            info = self.metal_tessellation_patch_info(
                self.parameter_raw_type(parameter)
            )
            if info is None or not info["valid"]:
                continue
            patch_descriptions.append(
                f"{info['kind']}:{parameter.name}->{info['element_type']}"
                f"[{info['count_text']}]"
            )
        if patch_descriptions:
            lines.append(
                "// CrossGL tessellation patch parameters: "
                f"{', '.join(patch_descriptions)}\n"
            )
        return "".join(lines)

    def metal_local_size_threadgroup_total(self, execution_config):
        values = [str(value).strip() for value in compute_local_size(execution_config)]
        literal_product = 1
        for value in values:
            try:
                literal_product *= int(value, 0)
            except ValueError:
                return " * ".join(f"({item})" for item in values)
        return str(literal_product)

    def metal_mesh_stage_output_config(
        self, shader_type, func, function_name, reserved_parameter_names
    ):
        if shader_type != "mesh":
            return None

        role_parameters = self.metal_mesh_output_role_parameters(func)
        if role_parameters:
            return self.explicit_metal_mesh_stage_output_config(
                func, function_name, reserved_parameter_names, role_parameters
            )

        max_vertices = self.metal_single_stage_attribute_argument(func, "max_vertices")
        max_primitives = self.metal_single_stage_attribute_argument(
            func, "max_primitives"
        )
        topology = self.metal_single_stage_attribute_argument(func, "outputtopology")
        if not (max_vertices and max_primitives and topology):
            return None

        vertex_type = self.unique_metal_generated_name(
            f"_CrossGLMetalMeshVertex_{function_name}",
            set(getattr(self, "structs_by_name", {})),
        )
        parameter_name = self.unique_metal_generated_name(
            "_crossglMeshOut", reserved_parameter_names
        )
        return {
            "parameter_name": parameter_name,
            "vertex_type": vertex_type,
            "max_vertices": max_vertices,
            "max_primitives": max_primitives,
            "topology": self.metal_mesh_topology(topology),
            "primitive_type": "void",
            "output_parameters": {},
            "generated_vertex_struct": True,
        }

    def explicit_metal_mesh_stage_output_config(
        self, func, function_name, reserved_parameter_names, role_parameters
    ):
        missing_roles = {"vertices", "indices"} - set(role_parameters)
        if missing_roles:
            missing = ", ".join(sorted(missing_roles))
            raise ValueError(
                "Metal mesh output parameters require explicit "
                f"{missing} output role(s)"
            )

        topology = self.metal_single_stage_attribute_argument(func, "outputtopology")
        if not topology:
            raise ValueError(
                "Metal mesh output parameters require an outputtopology attribute"
            )
        topology = self.metal_mesh_topology(topology)

        vertices = self.metal_mesh_output_array_parameter_info(
            role_parameters["vertices"], "vertices"
        )
        indices = self.metal_mesh_output_array_parameter_info(
            role_parameters["indices"], "indices"
        )
        primitives = None
        if "primitives" in role_parameters:
            primitives = self.metal_mesh_output_array_parameter_info(
                role_parameters["primitives"], "primitives"
            )
            if primitives["count"] != indices["count"]:
                raise ValueError(
                    "Metal mesh primitives output count must match indices output count"
                )

        self.validate_metal_mesh_indices_output_type(indices, topology)
        parameter_name = self.unique_metal_generated_name(
            "_crossglMeshOut", reserved_parameter_names
        )
        output_parameters = {
            vertices["name"]: {**vertices, "role": "vertices"},
            indices["name"]: {**indices, "role": "indices"},
        }
        primitive_type = "void"
        if primitives is not None:
            primitive_type = primitives["element_type"]
            output_parameters[primitives["name"]] = {
                **primitives,
                "role": "primitives",
            }

        return {
            "parameter_name": parameter_name,
            "vertex_type": vertices["element_type"],
            "primitive_type": primitive_type,
            "max_vertices": vertices["count"],
            "max_primitives": indices["count"],
            "topology": topology,
            "output_parameters": output_parameters,
            "generated_vertex_struct": False,
            "function_name": function_name,
        }

    def metal_mesh_output_role_parameters(self, func):
        role_parameters = {}
        for parameter in getattr(func, "parameters", []) or []:
            role = self.metal_mesh_output_parameter_role(parameter)
            if role is None:
                continue
            if role in role_parameters:
                raise ValueError(
                    f"Metal mesh output role '{role}' can be used by only one parameter"
                )
            role_parameters[role] = parameter
        return role_parameters

    def is_metal_mesh_output_parameter(self, shader_type, parameter):
        return (
            shader_type == "mesh"
            and self.metal_mesh_output_parameter_role(parameter) is not None
        )

    def metal_mesh_output_parameter_role(self, parameter):
        for attr in getattr(parameter, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower()
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized in {"vertices", "indices", "primitives"}:
                return normalized
        return None

    def metal_mesh_output_array_parameter_info(self, parameter, role):
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if hasattr(raw_type, "element_type") and hasattr(raw_type, "size"):
            element_type = self.map_type(self.type_name_string(raw_type.element_type))
            count = self.metal_mesh_output_array_size(raw_type.size)
        else:
            type_name = self.type_name_string(raw_type)
            element_type, array_suffix = split_array_type_suffix(type_name)
            element_type = self.map_type(element_type)
            count = self.metal_mesh_output_array_suffix_size(array_suffix)

        if not element_type or count is None:
            raise ValueError(f"Metal mesh {role} output must be a fixed-size array")
        return {
            "name": parameter.name,
            "element_type": element_type,
            "count": count,
        }

    def metal_mesh_output_array_size(self, size):
        if size is None:
            return None
        if isinstance(size, int):
            return str(size)
        return self.attribute_value_to_string(size)

    def metal_mesh_output_array_suffix_size(self, array_suffix):
        if not array_suffix:
            return None
        if not (array_suffix.startswith("[") and array_suffix.endswith("]")):
            return None
        size = array_suffix[1:-1].strip()
        return size or None

    def validate_metal_mesh_indices_output_type(self, indices, topology):
        expected_width = {"point": 1, "line": 2, "triangle": 3}[topology]
        expected_type = "uint" if expected_width == 1 else f"uint{expected_width}"
        actual_type = indices["element_type"]
        if actual_type != expected_type:
            raise ValueError(
                "Metal mesh indices output for "
                f"{topology} topology requires {expected_type}, got {actual_type}"
            )

    def validate_metal_mesh_output_usage(self, func):
        mesh_output = self.current_metal_mesh_output_config
        if mesh_output is None:
            return

        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        self.validate_metal_mesh_output_statement_sequence(
            statements,
            self.metal_mesh_output_flow_state(),
        )

    def metal_mesh_output_flow_state(
        self,
        counts_seen=False,
        counts=None,
        falls_through=True,
        terminator=None,
        exits=None,
    ):
        normalized_counts = None
        if counts is not None:
            normalized_counts = {
                "vertices": counts.get("vertices"),
                "primitives": counts.get("primitives"),
            }
        return {
            "counts_seen": counts_seen,
            "counts": normalized_counts,
            "falls_through": falls_through,
            "terminator": None if falls_through else terminator,
            "exits": list(exits or []),
        }

    def copy_metal_mesh_output_flow_state(
        self, state, falls_through=None, terminator=None, exits=None
    ):
        copied_falls_through = (
            state.get("falls_through", True) if falls_through is None else falls_through
        )
        return self.metal_mesh_output_flow_state(
            counts_seen=state.get("counts_seen", False),
            counts=state.get("counts"),
            falls_through=copied_falls_through,
            terminator=(
                None
                if copied_falls_through
                else (state.get("terminator") if terminator is None else terminator)
            ),
            exits=(list(state.get("exits", [])) if exits is None else exits),
        )

    def metal_mesh_output_exit_state(self, state, terminator=None):
        return self.metal_mesh_output_flow_state(
            counts_seen=state.get("counts_seen", False),
            counts=state.get("counts"),
            falls_through=False,
            terminator=(
                terminator if terminator is not None else state.get("terminator")
            ),
        )

    def merge_metal_mesh_output_flow_states(self, states, fallthrough_terminators=()):
        fallthrough_states = []
        propagated_exits = []
        for state in states or []:
            propagated_exits.extend(state.get("exits", []))
            if (
                state.get("falls_through", True)
                or state.get("terminator") in fallthrough_terminators
            ):
                fallthrough_states.append(
                    self.copy_metal_mesh_output_flow_state(
                        state,
                        falls_through=True,
                        exits=[],
                    )
                )
            elif state.get("terminator") is not None:
                propagated_exits.append(self.metal_mesh_output_exit_state(state))
        if not fallthrough_states:
            return self.metal_mesh_output_flow_state(
                falls_through=False,
                exits=propagated_exits,
            )

        counts_seen = all(
            state.get("counts_seen", False) for state in fallthrough_states
        )
        counts = None
        if counts_seen:
            counts = {}
            for role in ("vertices", "primitives"):
                role_values = []
                role_has_unknown = False
                for state in fallthrough_states:
                    state_counts = state.get("counts")
                    if state_counts is None or state_counts.get(role) is None:
                        role_has_unknown = True
                        break
                    role_values.append(state_counts[role])
                counts[role] = (
                    None if role_has_unknown or not role_values else min(role_values)
                )

        return self.metal_mesh_output_flow_state(
            counts_seen=counts_seen,
            counts=counts,
            falls_through=True,
            exits=propagated_exits,
        )

    def metal_mesh_output_states_for_terminator(self, state, terminator):
        states = [
            exit_state
            for exit_state in state.get("exits", [])
            if exit_state.get("terminator") == terminator
        ]
        if (
            not state.get("falls_through", True)
            and state.get("terminator") == terminator
        ):
            states.append(self.metal_mesh_output_exit_state(state, terminator))
        return states

    def metal_mesh_statement_list(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        return [body]

    def validate_metal_mesh_output_statement_sequence(self, statements, state):
        current = self.copy_metal_mesh_output_flow_state(state)
        for stmt in statements or []:
            if not current.get("falls_through", True):
                break
            current = self.validate_metal_mesh_output_statement_flow(stmt, current)
        return current

    def validate_metal_mesh_output_statement_flow(self, stmt, state):
        current = self.copy_metal_mesh_output_flow_state(state)
        if isinstance(stmt, IfNode):
            return self.validate_metal_mesh_if_output_flow(stmt, current)
        if isinstance(stmt, BlockNode):
            return self.validate_metal_mesh_output_statement_sequence(
                self.metal_mesh_statement_list(stmt),
                current,
            )
        if isinstance(stmt, SwitchNode):
            return self.validate_metal_mesh_switch_output_flow(stmt, current)
        if isinstance(stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)):
            return self.validate_metal_mesh_loop_output_flow(stmt, current)
        if isinstance(stmt, ReturnNode):
            return self.copy_metal_mesh_output_flow_state(
                current, falls_through=False, terminator="return"
            )
        if isinstance(stmt, BreakNode):
            return self.copy_metal_mesh_output_flow_state(
                current, falls_through=False, terminator="break"
            )
        if isinstance(stmt, ContinueNode):
            return self.copy_metal_mesh_output_flow_state(
                current, falls_through=False, terminator="continue"
            )

        expr = getattr(stmt, "expression", stmt)
        if isinstance(expr, IfNode):
            return self.validate_metal_mesh_if_output_flow(expr, current)
        if isinstance(expr, SwitchNode):
            return self.validate_metal_mesh_switch_output_flow(expr, current)

        mesh_counts = self.metal_mesh_set_output_counts(expr)
        if mesh_counts is not None:
            return self.metal_mesh_output_flow_state(
                counts_seen=True,
                counts=mesh_counts,
                falls_through=current.get("falls_through", True),
                exits=current.get("exits", []),
            )

        if isinstance(expr, AssignmentNode):
            self.validate_metal_mesh_output_assignment_usage(
                expr,
                current.get("counts_seen", False),
                current.get("counts"),
            )
            return current

        if isinstance(expr, FunctionCallNode):
            self.validate_metal_mesh_output_helper_call_usage(
                expr,
                current.get("counts_seen", False),
                current.get("counts"),
            )
            return current

        nested_statements = self.metal_mesh_nested_statements(stmt)
        for nested in nested_statements:
            self.validate_metal_mesh_output_statement_sequence(nested, current)

        return current

    def validate_metal_mesh_if_output_flow(self, stmt, state):
        then_state = self.validate_metal_mesh_output_statement_sequence(
            self.metal_mesh_statement_list(getattr(stmt, "then_branch", None)),
            state,
        )
        else_branch = getattr(stmt, "else_branch", None)
        if else_branch is None:
            else_state = self.copy_metal_mesh_output_flow_state(state)
        else:
            else_state = self.validate_metal_mesh_output_statement_sequence(
                self.metal_mesh_statement_list(else_branch),
                state,
            )
        return self.merge_metal_mesh_output_flow_states([then_state, else_state])

    def validate_metal_mesh_switch_output_flow(self, stmt, state):
        branch_states = []
        has_default = False
        for case in getattr(stmt, "cases", []) or []:
            if getattr(case, "value", None) is None:
                has_default = True
            branch_states.append(
                self.validate_metal_mesh_output_statement_sequence(
                    self.metal_mesh_statement_list(getattr(case, "statements", [])),
                    state,
                )
            )

        default_case = getattr(stmt, "default_case", None)
        if default_case is not None:
            has_default = True
            branch_states.append(
                self.validate_metal_mesh_output_statement_sequence(
                    self.metal_mesh_statement_list(default_case),
                    state,
                )
            )

        if not has_default:
            branch_states.append(self.copy_metal_mesh_output_flow_state(state))
        return self.merge_metal_mesh_output_flow_states(
            branch_states,
            fallthrough_terminators={"break"},
        )

    def validate_metal_mesh_loop_output_flow(self, stmt, state):
        body_state = self.validate_metal_mesh_output_statement_sequence(
            self.metal_mesh_statement_list(getattr(stmt, "body", [])),
            state,
        )
        break_states = self.metal_mesh_output_states_for_terminator(body_state, "break")
        continue_states = self.metal_mesh_output_states_for_terminator(
            body_state, "continue"
        )
        return_states = self.metal_mesh_output_states_for_terminator(
            body_state, "return"
        )

        loop_exits = list(return_states)
        if isinstance(stmt, LoopNode) or self.metal_mesh_loop_condition_is_true(stmt):
            fallthrough_states = break_states
        elif isinstance(stmt, DoWhileNode):
            fallthrough_states = list(break_states)
            if body_state.get("falls_through", True):
                fallthrough_states.append(
                    self.copy_metal_mesh_output_flow_state(body_state, exits=[])
                )
            fallthrough_states.extend(continue_states)
        else:
            fallthrough_states = [self.copy_metal_mesh_output_flow_state(state)]
            if body_state.get("falls_through", True):
                fallthrough_states.append(
                    self.copy_metal_mesh_output_flow_state(body_state, exits=[])
                )
            fallthrough_states.extend(break_states)
            fallthrough_states.extend(continue_states)

        if not fallthrough_states:
            return self.metal_mesh_output_flow_state(
                falls_through=False,
                exits=loop_exits,
            )

        merged = self.merge_metal_mesh_output_flow_states(
            fallthrough_states,
            fallthrough_terminators={"break", "continue"},
        )
        return self.copy_metal_mesh_output_flow_state(
            merged,
            exits=loop_exits + merged.get("exits", []),
        )

    def metal_mesh_loop_condition_is_true(self, stmt):
        if isinstance(stmt, LoopNode):
            return True
        if isinstance(stmt, ForNode) and getattr(stmt, "condition", None) is None:
            return True

        condition = getattr(stmt, "condition", None)
        literal_value = self.metal_literal_bool_value(condition)
        return literal_value is True

    def metal_literal_bool_value(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return None
        if hasattr(value, "value"):
            return self.metal_literal_bool_value(getattr(value, "value"))
        return None

    def metal_mesh_set_output_counts(self, expr):
        if not isinstance(expr, MeshOpNode) or expr.operation != "SetMeshOutputCounts":
            return None
        if len(expr.arguments) != 2:
            raise ValueError(
                "Metal mesh SetMeshOutputCounts requires exactly 2 arguments"
            )

        vertex_count = self.literal_int_value(
            expr.arguments[0], self.literal_int_constants
        )
        primitive_count = self.literal_int_value(
            expr.arguments[1], self.literal_int_constants
        )
        self.validate_metal_mesh_output_count_bound(
            "numVertices",
            vertex_count,
            self.current_metal_mesh_output_config["max_vertices"],
        )
        self.validate_metal_mesh_output_count_bound(
            "numPrimitives",
            primitive_count,
            self.current_metal_mesh_output_config["max_primitives"],
        )
        return {
            "vertices": vertex_count,
            "primitives": primitive_count,
        }

    def validate_metal_mesh_output_count_bound(self, label, value, declared_bound):
        if value is None:
            return
        declared = self.metal_literal_int_text_value(declared_bound)
        if declared is None:
            return
        if value > declared:
            raise ValueError(
                f"Metal mesh SetMeshOutputCounts {label} argument "
                f"({value}) cannot exceed declared output count ({declared})"
            )

    def validate_metal_mesh_output_assignment_usage(
        self, assignment, set_counts_seen, set_counts
    ):
        target = getattr(assignment, "target", getattr(assignment, "left", None))
        target_info = self.metal_mesh_output_assignment_target(target)
        if target_info is None:
            return

        role = target_info["role"]
        name = target_info["name"]
        if not set_counts_seen:
            raise ValueError(
                f"Metal mesh output array '{name}' must be written only after "
                "SetMeshOutputCounts"
            )

        index_value = self.literal_int_value(
            target_info.get("index"), self.literal_int_constants
        )
        if index_value is None:
            return

        declared_bound = self.metal_literal_int_text_value(
            target_info["output"]["count"]
        )
        if declared_bound is not None and index_value >= declared_bound:
            raise ValueError(
                f"Metal mesh {role} output array '{name}' index ({index_value}) "
                f"must be less than declared array size ({declared_bound})"
            )

        active_bound = None
        if set_counts is not None:
            active_bound = (
                set_counts.get("vertices")
                if role == "vertices"
                else set_counts.get("primitives")
            )
        if active_bound is not None and index_value >= active_bound:
            raise ValueError(
                f"Metal mesh {role} output array '{name}' index ({index_value}) "
                f"must be less than SetMeshOutputCounts "
                f"{'numVertices' if role == 'vertices' else 'numPrimitives'} "
                f"({active_bound})"
            )

    def validate_metal_mesh_output_helper_call_usage(
        self, call, set_counts_seen, set_counts
    ):
        func_name = self.function_call_name(call)
        if func_name == "SetIndex":
            self.validate_metal_mesh_set_index_helper_call_usage(
                call, set_counts_seen, set_counts
            )
            return

        helper_roles = {
            "SetVertex": ("vertices", "max_vertices", "numVertices"),
            "SetPrimitive": ("primitives", "max_primitives", "numPrimitives"),
        }
        role_info = helper_roles.get(func_name)
        if role_info is None:
            return

        args = getattr(call, "arguments", getattr(call, "args", []))
        if len(args) != 2:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' requires exactly 2 arguments"
            )

        if not set_counts_seen:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' must be called only after "
                "SetMeshOutputCounts"
            )

        index_value = self.literal_int_value(args[0], self.literal_int_constants)
        if index_value is None:
            return

        role, declared_key, active_label = role_info
        mesh_output = self.current_metal_mesh_output_config or {}
        declared_bound = self.metal_literal_int_text_value(
            mesh_output.get(declared_key)
        )
        if declared_bound is not None and index_value >= declared_bound:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' index ({index_value}) "
                f"must be less than declared output count ({declared_bound})"
            )

        active_bound = None
        if set_counts is not None:
            active_bound = set_counts.get(role)
        if active_bound is not None and index_value >= active_bound:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' index ({index_value}) "
                f"must be less than SetMeshOutputCounts {active_label} "
                f"({active_bound})"
            )

    def validate_metal_mesh_set_index_helper_call_usage(
        self, call, set_counts_seen, set_counts
    ):
        func_name = self.function_call_name(call)
        args = getattr(call, "arguments", getattr(call, "args", []))
        if len(args) != 2:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' requires exactly 2 arguments"
            )

        if not set_counts_seen:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' must be called only after "
                "SetMeshOutputCounts"
            )

        index_value = self.literal_int_value(args[0], self.literal_int_constants)
        if index_value is None:
            return

        write_width = self.metal_mesh_index_vector_width(args[1]) or 1
        mesh_output = self.current_metal_mesh_output_config or {}
        topology_width = self.metal_mesh_topology_index_width(
            mesh_output.get("topology")
        )
        if topology_width is None:
            return

        declared_primitives = self.metal_literal_int_text_value(
            mesh_output.get("max_primitives")
        )
        if declared_primitives is not None:
            declared_index_count = declared_primitives * topology_width
            self.validate_metal_mesh_set_index_span(
                func_name,
                index_value,
                write_width,
                declared_index_count,
                "declared flattened index count",
            )

        active_primitives = None
        if set_counts is not None:
            active_primitives = set_counts.get("primitives")
        if active_primitives is not None:
            active_index_count = active_primitives * topology_width
            self.validate_metal_mesh_set_index_span(
                func_name,
                index_value,
                write_width,
                active_index_count,
                "SetMeshOutputCounts numPrimitives flattened index count",
            )

    def validate_metal_mesh_set_index_span(
        self, func_name, index_value, write_width, flattened_bound, bound_label
    ):
        last_index = index_value + write_width - 1
        if index_value < 0:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' index ({index_value}) "
                "must be non-negative"
            )
        if last_index >= flattened_bound:
            raise ValueError(
                f"Metal mesh output helper '{func_name}' index range "
                f"[{index_value}, {last_index}] must be less than "
                f"{bound_label} ({flattened_bound})"
            )

    def metal_mesh_nested_statements(self, stmt):
        nested = []
        for attribute in (
            "body",
            "then_branch",
            "if_body",
            "else_branch",
            "else_body",
        ):
            value = getattr(stmt, attribute, None)
            if value is None:
                continue
            nested.append(
                getattr(value, "statements", value if isinstance(value, list) else [])
            )
        return [statements for statements in nested if statements]

    def metal_literal_int_text_value(self, value):
        if value is None:
            return None
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            return None

    def collect_function_metal_mesh_dispatch_contexts(self, functions):
        functions = list(functions)
        functions_by_name = {
            func.name: func for func in functions if getattr(func, "name", None)
        }
        contexts_by_id = {
            id(func): self.direct_metal_mesh_dispatch_context(func)
            for func in functions
        }

        changed = True
        while changed:
            changed = False
            for func in functions:
                name = getattr(func, "name", None)
                if not name:
                    continue
                context = contexts_by_id.setdefault(id(func), {})
                for called_name in self.called_user_function_names(func):
                    called_func = functions_by_name.get(called_name)
                    called_context = (
                        contexts_by_id.get(id(called_func))
                        if called_func is not None
                        else None
                    )
                    if not called_context:
                        continue
                    if self.merge_metal_mesh_dispatch_context(
                        context, called_context, name, called_name
                    ):
                        changed = True

        self.function_metal_mesh_dispatch_contexts_by_id = {
            func_id: context for func_id, context in contexts_by_id.items() if context
        }
        return {
            getattr(func, "name", None): context
            for func in functions
            for context in [contexts_by_id.get(id(func), {})]
            if getattr(func, "name", None) and context
        }

    def direct_metal_mesh_dispatch_context(self, func):
        context = {}
        if self.function_contains_mesh_op(
            func, "DispatchMesh", argument_counts={1, 3, 4}
        ):
            context["grid"] = True
        if not self.function_contains_mesh_op(
            func, "DispatchMesh", argument_counts={4}
        ):
            return context

        dispatch_payload_type = self.metal_dispatch_mesh_payload_argument_type(func)
        expected_payload_type = self.metal_program_mesh_payload_type
        if (
            dispatch_payload_type is not None
            and expected_payload_type is not None
            and dispatch_payload_type != expected_payload_type
        ):
            raise ValueError(
                "Metal DispatchMesh payload type "
                f"'{dispatch_payload_type}' must match mesh payload type "
                f"'{expected_payload_type}'"
            )

        payload_type = expected_payload_type or dispatch_payload_type
        if payload_type is None:
            raise ValueError("Metal DispatchMesh payload argument type is unknown")
        context["payload"] = True
        context["payload_type"] = payload_type
        return context

    def merge_metal_mesh_dispatch_context(
        self, context, called_context, func_name, called_name
    ):
        changed = False
        if called_context.get("grid") and not context.get("grid"):
            context["grid"] = True
            changed = True
        if called_context.get("payload"):
            if not context.get("payload"):
                context["payload"] = True
                changed = True
            payload_type = context.get("payload_type")
            called_payload_type = called_context.get("payload_type")
            if (
                payload_type
                and called_payload_type
                and payload_type != called_payload_type
            ):
                raise ValueError(
                    "Metal mesh dispatch helper payload type mismatch: "
                    f"'{func_name}' calls '{called_name}' requiring "
                    f"'{called_payload_type}' but already requires '{payload_type}'"
                )
            if called_payload_type and payload_type != called_payload_type:
                context["payload_type"] = called_payload_type
                changed = True
        return changed

    def metal_mesh_dispatch_context_for_function(self, func_or_name):
        if not isinstance(func_or_name, str):
            context = self.function_metal_mesh_dispatch_contexts_by_id.get(
                id(func_or_name)
            )
            if context is not None:
                return context
        name = (
            getattr(func_or_name, "name", None)
            if not isinstance(func_or_name, str)
            else func_or_name
        )
        if not name:
            return {}
        return self.function_metal_mesh_dispatch_contexts.get(name, {})

    def append_metal_mesh_dispatch_context_parameters(
        self, params_str, func, reserved_parameter_names
    ):
        context = self.metal_mesh_dispatch_context_for_function(func)
        if not context:
            return params_str

        reserved_names = set(reserved_parameter_names or ())
        reserved_names.update(self.metal_function_local_variable_names(func))
        if context.get("payload"):
            payload_type = (
                context.get("payload_type") or self.metal_program_mesh_payload_type
            )
            payload_name = self.unique_metal_generated_name(
                "_crossglMeshPayload", reserved_names
            )
            reserved_names.add(payload_name)
            params_str = self.append_parameter_declaration(
                params_str,
                f"object_data {payload_type}& {payload_name}",
            )
            self.current_metal_mesh_payload_parameter = payload_name
            self.current_metal_mesh_payload_type = payload_type
            self.current_address_space_variables[payload_name] = "object_data"

        if context.get("grid"):
            grid_name = self.unique_metal_generated_name(
                "_crossglMeshGrid", reserved_names
            )
            params_str = self.append_parameter_declaration(
                params_str,
                f"mesh_grid_properties {grid_name}",
            )
            self.current_metal_mesh_grid_properties_parameter = grid_name
        return params_str

    def required_metal_mesh_dispatch_context_arguments(self, func_name):
        context = self.metal_mesh_dispatch_context_for_function(func_name)
        if not context:
            return []
        args = []
        if context.get("payload"):
            args.append(self.current_metal_mesh_payload_parameter)
        if context.get("grid"):
            args.append(self.current_metal_mesh_grid_properties_parameter)
        return [arg for arg in args if arg]

    def metal_mesh_dispatch_context_call_diagnostic(self, func_name):
        context = self.metal_mesh_dispatch_context_for_function(func_name)
        if not context:
            return None
        if context.get("payload") and not self.current_metal_mesh_payload_parameter:
            return (
                "/* unsupported Metal mesh dispatch helper call: function "
                f"'{func_name}' requires an object_data payload context */"
            )
        if (
            context.get("grid")
            and not self.current_metal_mesh_grid_properties_parameter
        ):
            return (
                "/* unsupported Metal mesh dispatch helper call: function "
                f"'{func_name}' requires mesh_grid_properties context */"
            )
        return None

    def metal_mesh_payload_type_for_program(self, ast):
        payload_types = set()
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            if normalize_stage_name(stage_type) != "mesh":
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            for parameter in getattr(entry_point, "parameters", []) or []:
                if not self.is_metal_mesh_payload_parameter("mesh", parameter):
                    continue
                payload_types.add(self.metal_parameter_mapped_type(parameter))

        if len(payload_types) > 1:
            expected = ", ".join(sorted(payload_types))
            raise ValueError(
                "Metal mesh stages in one pipeline must use one mesh payload type; "
                f"got {expected}"
            )
        return next(iter(payload_types), None)

    def validate_metal_mesh_payload_parameter_placement(self, ast, functions):
        mesh_payload_entry_stages_by_id = {}
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in {"object", "task", "amplification", "mesh"}:
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                mesh_payload_entry_stages_by_id[id(entry_point)] = stage_name
                for parameter in getattr(entry_point, "parameters", []) or []:
                    if self.is_metal_mesh_payload_parameter(stage_name, parameter):
                        self.validate_metal_mesh_payload_parameter_address_space(
                            stage_name, parameter
                        )

        for func in functions:
            if id(func) in mesh_payload_entry_stages_by_id:
                continue
            for parameter in getattr(func, "parameters", []) or []:
                if not self.is_explicit_metal_mesh_payload_semantic(
                    self.semantic_from_node(parameter)
                ):
                    continue
                func_name = getattr(func, "name", "<anonymous>")
                param_name = getattr(parameter, "name", "<anonymous>")
                raise ValueError(
                    "Metal mesh payload parameters are only supported on "
                    "object, task, amplification, or mesh stage entry points; "
                    f"function '{func_name}' parameter '{param_name}' declares "
                    "mesh payload semantics"
                )

    def validate_metal_mesh_payload_parameter_address_space(
        self, stage_name, parameter
    ):
        address_space = self.normalized_address_space(
            self.parameter_address_space(parameter)
        )
        if address_space in {None, "object_data"}:
            return

        param_name = getattr(parameter, "name", "<anonymous>")
        raise ValueError(
            f"Metal mesh payload parameter '{param_name}' on {stage_name} stage "
            f"uses {address_space} address space; mesh payload parameters "
            "require object_data"
        )

    def generated_metal_mesh_payload_parameter(
        self, shader_type, func, reserved_parameter_names
    ):
        if shader_type not in {"object", "task", "amplification"}:
            return None
        if self.current_metal_mesh_payload_parameter is not None:
            return None
        context = self.metal_mesh_dispatch_context_for_function(func)
        if not context.get("payload"):
            return None

        expected_payload_type = self.metal_program_mesh_payload_type
        dispatch_payload_type = context.get("payload_type")
        if (
            dispatch_payload_type is not None
            and expected_payload_type is not None
            and dispatch_payload_type != expected_payload_type
        ):
            raise ValueError(
                "Metal DispatchMesh payload type "
                f"'{dispatch_payload_type}' must match mesh payload type "
                f"'{expected_payload_type}'"
            )

        payload_type = expected_payload_type or dispatch_payload_type
        if payload_type is None:
            raise ValueError("Metal DispatchMesh payload argument type is unknown")

        reserved_names = set(reserved_parameter_names or ())
        reserved_names.update(self.metal_function_local_variable_names(func))
        payload_name = self.unique_metal_generated_name(
            "_crossglMeshPayload", reserved_names
        )
        return payload_type, payload_name

    def metal_dispatch_mesh_payload_argument_type(self, func):
        declared_types = self.metal_function_declared_value_types(func)
        payload_types = set()
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, MeshOpNode):
                continue
            arguments = self.normalized_metal_intrinsic_args(node.arguments)
            if node.operation != "DispatchMesh" or len(arguments) != 4:
                continue
            payload_type = self.metal_dispatch_mesh_payload_type(
                arguments[3], declared_types
            )
            if payload_type is not None:
                payload_types.add(payload_type)

        if len(payload_types) > 1:
            expected = ", ".join(sorted(payload_types))
            raise ValueError(
                "Metal DispatchMesh payload arguments must use one type; "
                f"got {expected}"
            )
        return next(iter(payload_types), None)

    def metal_dispatch_mesh_payload_type(self, expr, declared_types):
        if isinstance(expr, ArrayAccessNode):
            payload_type = self.metal_dispatch_mesh_payload_type(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                declared_types,
            )
            return self.metal_mesh_payload_value_type(payload_type)
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            object_type = self.metal_dispatch_mesh_payload_type(
                object_expr, declared_types
            )
            object_type = self.metal_mesh_payload_value_type(object_type)
            member_type = self.struct_member_types.get(object_type, {}).get(expr.member)
            if member_type is not None:
                return self.metal_mesh_payload_value_type(member_type)
        expr_name = self.expression_name(expr)
        if expr_name in declared_types:
            return self.metal_mesh_payload_value_type(declared_types[expr_name])
        payload_type = self.expression_result_type(expr)
        return self.metal_mesh_payload_value_type(payload_type)

    def metal_mesh_payload_value_type(self, payload_type):
        if not payload_type:
            return None
        payload_type = self.type_name_string(payload_type).strip()
        while payload_type.endswith(("&", "*")):
            payload_type = payload_type[:-1].strip()
        if "[" in payload_type and "]" in payload_type:
            payload_type, _ = split_array_type_suffix(payload_type)
        return self.map_type(payload_type)

    def metal_function_declared_value_types(self, func):
        declared_types = {}
        for parameter in getattr(func, "parameters", []) or []:
            declared_types[parameter.name] = self.metal_parameter_raw_type(parameter)
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            raw_type = getattr(node, "var_type", getattr(node, "vtype", None))
            if raw_type is not None:
                declared_types[name] = self.type_name_string(raw_type)
        return declared_types

    def metal_function_local_variable_names(self, func):
        names = set()
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                names.add(node.name)
        return names

    def collect_metal_local_identifier_remaps(
        self, func, parameters, stage_parameters, stage_local_variables=None
    ):
        reserved_names = []
        for parameter in parameters or []:
            name = getattr(parameter, "name", None)
            if name:
                reserved_names.append(name)

        ordered_names = []
        for parameter in stage_parameters or []:
            name = parameter.get("name") if isinstance(parameter, dict) else None
            if name:
                ordered_names.append(name)
        for variable in stage_local_variables or []:
            name = getattr(variable, "name", None)
            if name:
                ordered_names.append(name)
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                ordered_names.append(node.name)

        used_names = set(reserved_names) | set(ordered_names)
        remaps = {}
        for name in ordered_names:
            if name in remaps or name not in self.METAL_RESERVED_LOCAL_IDENTIFIERS:
                continue
            base_name = f"{name}_"
            candidate = base_name
            suffix = 2
            while candidate in used_names:
                candidate = f"{base_name}{suffix}"
                suffix += 1
            remaps[name] = candidate
            used_names.add(candidate)
        return remaps

    def metal_local_identifier_name(self, name):
        return self.current_local_identifier_remaps.get(name, name)

    def metal_parameter_mapped_type(self, parameter):
        return self.map_type(self.metal_parameter_raw_type(parameter))

    def metal_parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return self.type_name_string(parameter.param_type)
        if hasattr(parameter, "vtype"):
            return self.type_name_string(parameter.vtype)
        return "float"

    def metal_mesh_grid_properties_parameter(
        self, shader_type, func, reserved_parameter_names
    ):
        if shader_type not in {"object", "task", "amplification"}:
            return None
        context = self.metal_mesh_dispatch_context_for_function(func)
        if not context.get("grid"):
            return None
        return self.unique_metal_generated_name(
            "_crossglMeshGrid", reserved_parameter_names
        )

    def function_contains_mesh_op(self, func, operation, argument_counts=None):
        for node in self.iter_ast_nodes(getattr(func, "body", None)):
            if not isinstance(node, MeshOpNode) or node.operation != operation:
                continue
            arguments = self.normalized_metal_intrinsic_args(node.arguments)
            if argument_counts is not None and len(arguments) not in argument_counts:
                continue
            return True
        return False

    def unique_metal_generated_name(self, base_name, reserved_names):
        reserved_names = set(reserved_names or ())
        if base_name not in reserved_names:
            return base_name
        index = 1
        while f"{base_name}_{index}" in reserved_names:
            index += 1
        return f"{base_name}_{index}"

    def metal_single_stage_attribute_argument(self, func, attribute_name):
        arguments = self.metal_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"Metal stage attribute {attribute_name} requires exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def metal_mesh_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise ValueError(
                f"Metal mesh output topology cannot be lowered: {topology}"
            )
        return mapped

    def metal_mesh_topology_index_width(self, topology):
        return {"point": 1, "line": 2, "triangle": 3}.get(topology)

    def generate_metal_mesh_vertex_output_struct(self, mesh_output):
        return (
            f"struct {mesh_output['vertex_type']} {{\n"
            "    float4 position [[position]];\n"
            "};\n"
        )

    def metal_mesh_stage_output_parameter_declaration(self, mesh_output):
        return (
            f"mesh<{mesh_output['vertex_type']}, {mesh_output['primitive_type']}, "
            f"{mesh_output['max_vertices']}, {mesh_output['max_primitives']}, "
            f"topology::{mesh_output['topology']}> {mesh_output['parameter_name']}"
        )

    def append_parameter_declaration(self, params_str, declaration):
        if not params_str:
            return declaration
        return f"{params_str}, {declaration}"

    def should_wrap_metal_vertex_stage_input_parameter(
        self, raw_param_type, shader_type, parameter
    ):
        if shader_type != "vertex":
            return False
        if self.is_metal_resource_or_buffer_parameter(
            raw_param_type, parameter, shader_type
        ):
            return False
        if self.type_name_string(raw_param_type) in self.structs_by_name:
            return False
        if self.is_metal_ray_stage(shader_type):
            return False
        if self.is_graphics_stage_output_parameter(parameter, shader_type):
            return False
        return True

    def is_metal_resource_or_buffer_parameter(
        self, raw_param_type, parameter, shader_type=None
    ):
        if self.is_resource_parameter_type(raw_param_type):
            return True
        if self.is_raw_buffer_parameter_type(raw_param_type, parameter):
            return True
        if self.is_bound_scalar_value_parameter(raw_param_type, parameter, shader_type):
            return True
        return False

    def all_functions(self, ast):
        functions = list(getattr(ast, "functions", []) or [])
        for stage in getattr(ast, "stages", {}).values():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
            functions.extend(getattr(stage, "local_functions", []) or [])
        return functions

    def generate_metal_builtin_limit_fallbacks(self, ast, functions):
        name = "gl_MaxImageUnits"
        declared_names = {
            getattr(node, "name", None)
            for node in [
                *(getattr(ast, "constants", []) or []),
                *(getattr(ast, "global_variables", []) or []),
            ]
        }
        if name in declared_names or not self.functions_use_identifier(functions, name):
            return ""
        default = GLSL_BUILTIN_INT_LIMITS[name]
        return (
            "/* CrossGL fallback: GLSL builtin limit specialization "
            "gl_MaxImageUnits is not available in Metal; using the OpenGL "
            "minimum value. */\n"
            f"{self.metal_unused_declaration_qualifier('constant')} "
            f"int gl_MaxImageUnits = {default};\n"
        )

    def functions_use_identifier(self, functions, name):
        return any(
            self.function_body_uses_identifier(getattr(func, "body", []), name)
            for func in functions or []
        )

    def collect_function_parameters(self, functions):
        parameters = []
        for func in functions or []:
            parameters.extend(getattr(func, "parameters", getattr(func, "params", [])))
        return parameters

    def collect_function_parameter_infos(self, functions):
        parameter_infos = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            infos = []
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                infos.append(
                    (getattr(param, "name", None), self.type_name_string(raw_type))
                )
            parameter_infos[func_name] = infos
        return parameter_infos

    def collect_function_parameter_nodes(self, functions):
        parameter_nodes = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            parameter_nodes[func_name] = list(
                getattr(func, "parameters", getattr(func, "params", [])) or []
            )
        return parameter_nodes

    def collect_unused_array_parameter_indices(self, functions):
        skipped = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            indices = set()
            body = getattr(func, "body", [])
            for index, param in enumerate(
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                param_name = getattr(param, "name", None)
                if not param_name:
                    continue
                raw_type = self.parameter_raw_type(param)
                if not self.is_array_type_node(raw_type):
                    continue
                if self.is_resource_parameter_type(raw_type):
                    continue
                if not self.function_body_uses_identifier(body, param_name):
                    indices.add(index)
            if indices:
                skipped[func_name] = indices
        return skipped

    def skipped_function_parameter_indices(self, func_name):
        return self.metal_skipped_function_parameter_indices.get(func_name, set())

    def function_body_uses_identifier(self, body, name):
        for node in self.iter_ast_nodes(body):
            if getattr(node, "name", None) == name:
                return True
        return False

    def structured_buffer_parameter_type_map(self, func):
        parameter_types = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])):
            name = getattr(param, "name", None)
            raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
            type_name = self.type_name_string(raw_type)
            if name and self.is_structured_buffer_type(type_name):
                parameter_types[name] = type_name
        return parameter_types

    def function_call_arguments(self, call):
        return getattr(call, "arguments", getattr(call, "args", []))

    def function_call_matches_known_signature(self, func_name, args):
        if func_name not in self.user_function_names:
            return False
        parameter_nodes = self.function_parameter_nodes.get(func_name)
        if parameter_nodes is None:
            return False
        return len(args or []) == len(parameter_nodes)

    def collect_function_structured_buffer_length_dependencies(self, functions):
        parameter_types = {
            getattr(func, "name", None): self.structured_buffer_parameter_type_map(func)
            for func in functions or []
            if getattr(func, "name", None)
        }
        dependencies = {func_name: set() for func_name in parameter_types}

        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for node in self.iter_ast_nodes(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                if self.function_call_name(node) != "buffer_dimensions":
                    continue
                args = self.function_call_arguments(node)
                if not args:
                    continue
                buffer_name = self.expression_name(args[0])
                if buffer_name in parameter_types.get(func_name, {}):
                    dependencies[func_name].add(buffer_name)

        changed = True
        while changed:
            changed = False
            for func in functions or []:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                current_dependencies = dependencies.setdefault(func_name, set())
                before = set(current_dependencies)
                for node in self.iter_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    called_name = self.function_call_name(node)
                    if called_name not in self.user_function_names:
                        continue
                    called_dependencies = dependencies.get(called_name, set())
                    if not called_dependencies:
                        continue
                    called_parameters = self.function_parameter_infos.get(
                        called_name, []
                    )
                    for index, arg in enumerate(self.function_call_arguments(node)):
                        if index >= len(called_parameters):
                            continue
                        called_param_name, _called_param_type = called_parameters[index]
                        if called_param_name not in called_dependencies:
                            continue
                        arg_name = self.expression_name(arg)
                        if arg_name in parameter_types.get(func_name, {}):
                            current_dependencies.add(arg_name)
                if current_dependencies != before:
                    changed = True
        return dependencies

    def collect_global_structured_buffer_length_dependencies(
        self, functions, global_vars
    ):
        global_buffer_names = set()
        for node in global_vars or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            raw_type = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_structured_buffer_type(raw_type):
                global_buffer_names.add(name)

        dependencies = set()
        for func in functions or []:
            local_names = {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if getattr(param, "name", None)
            }
            for node in self.iter_ast_nodes(getattr(func, "body", [])):
                if isinstance(node, VariableNode) and getattr(node, "name", None):
                    local_names.add(node.name)

            for node in self.iter_ast_nodes(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_name = self.function_call_name(node)
                args = self.function_call_arguments(node)
                if func_name == "buffer_dimensions" and args:
                    buffer_name = self.expression_name(args[0])
                    if (
                        buffer_name in global_buffer_names
                        and buffer_name not in local_names
                    ):
                        dependencies.add(buffer_name)
                    continue
                if func_name not in self.user_function_names:
                    continue
                called_dependencies = (
                    self.function_structured_buffer_length_dependencies.get(
                        func_name, set()
                    )
                )
                if not called_dependencies:
                    continue
                called_parameters = self.function_parameter_infos.get(func_name, [])
                for index, arg in enumerate(args):
                    if index >= len(called_parameters):
                        continue
                    called_param_name, _called_param_type = called_parameters[index]
                    if called_param_name not in called_dependencies:
                        continue
                    arg_name = self.expression_name(arg)
                    if arg_name in global_buffer_names and arg_name not in local_names:
                        dependencies.add(arg_name)
        return dependencies

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in getattr(root, "global_variables", []) or []:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and (
                self.is_resource_parameter_type(var_type)
                or self.glsl_buffer_block_attribute(node) is not None
            ):
                resource_names.add(var_name)
        return resource_names

    def is_stage_local_resource_variable(self, node):
        vtype = self.type_name_string(
            getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        return (
            self.metal_buffer_resource_address_space(node, vtype) is not None
            or self.is_resource_parameter_type(vtype)
            or self.is_glsl_buffer_block_variable(node, vtype)
        )

    def metal_buffer_resource_address_space(self, node, vtype=None):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if not qualifiers & {"uniform", "buffer"}:
            return None

        type_name = self.type_name_string(
            vtype if vtype is not None else getattr(node, "var_type", None)
        )
        if (
            self.is_resource_parameter_type(type_name)
            or self.is_structured_buffer_type(type_name)
            or self.is_glsl_buffer_block_variable(node, type_name)
        ):
            return None

        if "uniform" in qualifiers:
            return "constant"
        return "device"

    def metal_buffer_resource_binding_count(self, node, array_size=None):
        return 1

    def format_metal_buffer_resource_parameter(
        self, raw_type, name, array_size=None, address_space="device"
    ):
        mapped_type = self.map_resource_type_with_format(raw_type)
        if str(mapped_type).endswith("*"):
            if array_size is not None:
                array_size = array_size or "1"
                return f"array<{address_space} {mapped_type}, {array_size}> {name}"
            return f"{address_space} {mapped_type} {name}"
        if array_size is not None:
            return f"{address_space} {mapped_type}* {name}"
        return f"{address_space} {mapped_type}& {name}"

    def register_metal_buffer_resource_parameter_scope(
        self, func_name, include_all=False
    ):
        if include_all:
            resources = [
                (node, buffer_type, address_space)
                for node, _, buffer_type, _, address_space in (
                    self.metal_buffer_resource_variables
                )
            ]
        else:
            resources = [
                (node, buffer_type, address_space)
                for node, buffer_type, _, address_space in (
                    self.required_function_metal_buffer_resources(func_name)
                )
            ]

        for node, buffer_type, address_space in resources:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            if not name:
                continue
            self.local_variable_types[name] = self.type_name_string(buffer_type)
            self.current_address_space_variables[name] = address_space
            if address_space == "constant":
                self.current_readonly_metal_parameters.add(name)
                self.current_readonly_metal_parameter_reasons[name] = (
                    "constant address space"
                )

    def validate_global_resource_shadows(self, ast):
        conflicts = collect_non_resource_global_resource_shadows(
            ast,
            self.collect_global_resource_names(ast),
            self.is_resource_parameter_type,
        )
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Non-resource local declaration(s) shadow Metal global resource(s): "
                f"{names}"
            )

    def collect_function_cbuffer_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.direct_cbuffer_dependencies(func)
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True
        return dependencies

    def direct_cbuffer_dependencies(self, func):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)

        member_to_cbuffer = {}
        for cbuffer in self.cbuffer_variables:
            cbuffer_name = getattr(cbuffer, "name", None)
            if not cbuffer_name:
                continue
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                if member_name:
                    member_to_cbuffer[member_name] = cbuffer_name

        dependencies = set()
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not (hasattr(node, "__class__") and "Identifier" in str(node.__class__)):
                continue
            name = getattr(node, "name", None)
            if not name or name in local_names:
                continue
            cbuffer_name = member_to_cbuffer.get(name)
            if cbuffer_name:
                dependencies.add(cbuffer_name)
        return dependencies

    def called_user_function_names(self, func):
        called_names = set()
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_name(node)
            if func_name in self.user_function_names and func_name != getattr(
                func, "name", None
            ):
                called_names.add(func_name)
        return called_names

    def collect_function_stage_parameter_dependencies(self, ast, target_stage):
        direct_dependencies = {}
        function_calls = {}
        parameter_nodes = {}

        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if func_name:
                direct_dependencies.setdefault(func_name, set())
                function_calls.setdefault(
                    func_name, self.called_user_function_names(func)
                )

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            stage_parameters = {
                parameter.name: parameter
                for parameter in (
                    getattr(
                        entry_point, "parameters", getattr(entry_point, "params", [])
                    )
                    or []
                )
                if getattr(parameter, "name", None)
            }
            for parameter in stage_parameters.values():
                parameter_nodes[parameter.name] = parameter

            stage_functions = list(getattr(stage, "local_functions", []) or [])
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                direct_dependencies[func_name] = (
                    self.direct_stage_parameter_dependencies(func, stage_parameters)
                )
                function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True

        return {
            func_name: [
                parameter_nodes[name]
                for name in sorted(dependency_names)
                if name in parameter_nodes
            ]
            for func_name, dependency_names in dependencies.items()
            if dependency_names
        }

    def direct_stage_parameter_dependencies(self, func, stage_parameters):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)

        dependencies = set()
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not (hasattr(node, "__class__") and "Identifier" in str(node.__class__)):
                continue
            name = getattr(node, "name", None)
            if name and name in stage_parameters and name not in local_names:
                dependencies.add(name)
        return dependencies

    def required_function_stage_parameters(self, func_name):
        return self.function_stage_parameter_dependencies.get(func_name, [])

    def required_function_stage_parameter_argument_names(self, func_name):
        return [
            parameter.name
            for parameter in self.required_function_stage_parameters(func_name)
        ]

    def collect_function_stage_output_dependencies(self, ast, target_stage):
        direct_dependencies = {}
        function_calls = {}
        output_types = {}

        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if func_name:
                direct_dependencies.setdefault(func_name, None)
                function_calls.setdefault(
                    func_name, self.called_user_function_names(func)
                )

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            output_type = getattr(entry_point, "return_type", None)
            if output_type is None or self.type_name_string(output_type) == "void":
                continue
            stage_functions = list(getattr(stage, "local_functions", []) or [])
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                output_types[func_name] = output_type
                if self.direct_stage_output_dependency(func):
                    direct_dependencies[func_name] = output_type
                function_calls[func_name] = self.called_user_function_names(func)

        dependencies = dict(direct_dependencies)
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                if dependencies.get(func_name) is not None:
                    continue
                for called_name in calls:
                    if dependencies.get(called_name) is not None:
                        dependencies[func_name] = output_types.get(
                            func_name
                        ) or dependencies.get(called_name)
                        changed = True
                        break

        return {
            func_name: output_type
            for func_name, output_type in dependencies.items()
            if output_type is not None
        }

    def direct_stage_output_dependency(self, func):
        parameter_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
        }
        if "output" in parameter_names:
            return False
        local_names = set(parameter_names)
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)
        if "output" in local_names:
            return False
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, MemberAccessNode):
                if self.expression_name(getattr(node, "object", None)) == "output":
                    return True
            if getattr(node, "name", None) == "output":
                return True
        return False

    def required_function_stage_output_type(self, func_name):
        return self.function_stage_output_dependencies.get(func_name)

    def required_function_stage_output_argument_names(self, func_name):
        if self.required_function_stage_output_type(func_name) is None:
            return []
        return ["output"]

    def required_function_cbuffers(self, func_name):
        dependencies = self.function_cbuffer_dependencies.get(func_name, set())
        return [
            cbuffer
            for cbuffer in self.cbuffer_variables
            if getattr(cbuffer, "name", None) in dependencies
        ]

    def collect_function_global_resource_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.direct_global_resource_dependencies(
                func
            )
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True
        return dependencies

    def direct_global_resource_dependencies(self, func):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)

        texture_names = self.global_texture_names()
        acceleration_structure_names = self.global_acceleration_structure_names()
        visible_function_table_names = self.global_visible_function_table_names()
        intersection_function_table_names = (
            self.global_intersection_function_table_names()
        )
        buffer_names = (
            self.global_structured_buffer_names()
            | self.global_metal_buffer_resource_names()
            | self.global_glsl_buffer_block_names()
        )
        sampler_names = self.global_sampler_names()
        texture_alias_sources = self.local_texture_alias_sources(func, texture_names)
        acceleration_structure_alias_types = (
            self.local_acceleration_structure_dependency_types(
                func, acceleration_structure_names
            )
        )
        dependencies = set()

        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if hasattr(node, "__class__") and "Identifier" in str(node.__class__):
                name = getattr(node, "name", None)
                if (
                    name
                    and name not in local_names
                    and (
                        name in texture_names
                        or name in acceleration_structure_names
                        or name in visible_function_table_names
                        or name in intersection_function_table_names
                        or name in buffer_names
                        or name in sampler_names
                    )
                ):
                    dependencies.add(name)

            if isinstance(node, FunctionCallNode):
                self.add_texture_call_resource_dependencies(
                    node,
                    local_names,
                    texture_names,
                    sampler_names,
                    dependencies,
                    texture_alias_sources,
                )
                self.add_buffer_call_resource_dependencies(
                    node, local_names, buffer_names, dependencies
                )
            if isinstance(node, RayTracingOpNode):
                self.add_ray_tracing_resource_dependencies(
                    node,
                    local_names,
                    visible_function_table_names,
                    intersection_function_table_names,
                    acceleration_structure_alias_types,
                    dependencies,
                )

        return dependencies

    def local_acceleration_structure_dependency_types(
        self, func, acceleration_structure_names
    ):
        aliases = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])):
            param_name = getattr(param, "name", None)
            param_type = self.type_name_string(
                getattr(
                    param,
                    "param_type",
                    getattr(param, "var_type", getattr(param, "vtype", None)),
                )
            )
            if (
                param_name
                and param_type
                and self.is_acceleration_structure_type(param_type)
                and not self.metal_type_is_pointer_like(param_type)
            ):
                aliases[param_name] = self.map_resource_type_with_format(param_type)

        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            initial_type = self.acceleration_structure_dependency_argument_type(
                getattr(node, "initial_value", None),
                aliases,
                acceleration_structure_names,
            )
            if initial_type is None:
                continue

            declared_type = self.local_variable_type_node(node)
            declared_type_name = self.type_name_string(declared_type)
            if not declared_type_name:
                aliases[name] = initial_type
                continue
            if not self.is_acceleration_structure_type(declared_type_name):
                continue

            declared_resource_type = self.map_resource_type_with_format(
                declared_type_name, node
            )
            if declared_resource_type == initial_type:
                aliases[name] = declared_resource_type
        return aliases

    def acceleration_structure_dependency_argument_type(
        self, argument, alias_types=None, acceleration_structure_names=None
    ):
        if argument is None:
            return None

        argument_type = self.expression_result_type(argument)
        if argument_type and self.is_acceleration_structure_type(argument_type):
            if self.metal_type_is_pointer_like(argument_type):
                return None
            return self.map_resource_type_with_format(argument_type)

        argument_name = self.expression_name(argument)
        if not argument_name:
            return None
        if alias_types and argument_name in alias_types:
            return alias_types[argument_name]

        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            array_size,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) != argument_name:
                continue
            if (
                acceleration_structure_names is not None
                and argument_name not in acceleration_structure_names
            ):
                return None
            if array_size is not None and not isinstance(argument, ArrayAccessNode):
                return None
            return mapped_type
        return None

    def local_texture_alias_sources(self, func, texture_names):
        aliases = {}
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            initial_value = getattr(node, "initial_value", None)
            source_name = self.expression_name(initial_value)
            if (
                name
                and source_name
                and (source_name in texture_names or source_name in aliases)
            ):
                aliases[name] = initial_value
        return aliases

    def texture_alias_dependency_source(self, texture_name, texture_alias_sources):
        source_arg = texture_alias_sources.get(texture_name)
        seen = {texture_name}
        while source_arg is not None:
            source_name = self.expression_name(source_arg)
            if not source_name or source_name in seen:
                return source_arg
            next_source_arg = texture_alias_sources.get(source_name)
            if next_source_arg is None:
                return source_arg
            seen.add(source_name)
            source_arg = next_source_arg
        return None

    def add_ray_tracing_resource_dependencies(
        self,
        call,
        local_names,
        visible_function_table_names,
        intersection_function_table_names,
        acceleration_structure_alias_types,
        dependencies,
    ):
        operation = getattr(call, "operation", None)
        if operation == "TraceRay":
            args = getattr(call, "arguments", [])
            if not args:
                return
            acceleration_structure_type = (
                self.acceleration_structure_dependency_argument_type(
                    args[0], acceleration_structure_alias_types
                )
                or self.metal_acceleration_structure_argument_type(args[0])
            )
            table = self.default_intersection_function_table(
                acceleration_structure_type
            )
            if (
                table is not None
                and table["name"] in intersection_function_table_names
                and table["name"] not in local_names
            ):
                dependencies.add(table["name"])
            return

        if operation != "CallShader":
            return
        args = getattr(call, "arguments", [])
        if len(args) == 3:
            table_name = self.expression_name(args[0])
            if (
                table_name in visible_function_table_names
                and table_name not in local_names
            ):
                dependencies.add(table_name)
            return
        if len(args) == 2 and len(self.visible_function_table_variables) == 1:
            table_name = getattr(
                self.visible_function_table_variables[0][0], "name", None
            )
            if table_name and table_name not in local_names:
                dependencies.add(table_name)

    def add_texture_call_resource_dependencies(
        self,
        call,
        local_names,
        texture_names,
        sampler_names,
        dependencies,
        texture_alias_sources=None,
    ):
        func_name = self.function_call_name(call)
        if not func_name or not str(func_name).startswith(("texture", "image")):
            return
        args = getattr(call, "arguments", getattr(call, "args", []))
        if not args:
            return

        texture_name = self.expression_name(args[0])
        if texture_name in texture_names and texture_name not in local_names:
            dependencies.add(texture_name)

        if len(args) >= 3:
            sampler_name = self.expression_name(args[1])
            if sampler_name in sampler_names and sampler_name not in local_names:
                dependencies.add(sampler_name)
                return

        if not self.texture_call_needs_implicit_sampler_dependency(func_name, args):
            return
        implicit_sampler_name = f"{texture_name}Sampler" if texture_name else None
        if (
            implicit_sampler_name in sampler_names
            and implicit_sampler_name not in local_names
        ):
            dependencies.add(implicit_sampler_name)
            return

        source_arg = self.texture_alias_dependency_source(
            texture_name, texture_alias_sources or {}
        )
        source_name = self.expression_name(source_arg)
        source_sampler_name = f"{source_name}Sampler" if source_name else None
        if (
            source_sampler_name in sampler_names
            and source_sampler_name not in local_names
        ):
            dependencies.add(source_sampler_name)

    def add_buffer_call_resource_dependencies(
        self, call, local_names, buffer_names, dependencies
    ):
        func_name = self.function_call_name(call)
        if func_name not in {
            "buffer_load",
            "buffer_store",
            "buffer_append",
            "buffer_consume",
            "buffer_dimensions",
        }:
            return
        args = getattr(call, "arguments", getattr(call, "args", []))
        if not args:
            return
        buffer_name = self.expression_name(args[0])
        if buffer_name in buffer_names and buffer_name not in local_names:
            dependencies.add(buffer_name)

    def texture_sampling_uses_implicit_sampler(self, func_name):
        return is_texture_sampling_operation(
            func_name
        ) or is_texture_query_lod_operation(func_name)

    def texture_call_needs_implicit_sampler_dependency(self, func_name, args):
        if not self.texture_sampling_uses_implicit_sampler(func_name):
            return False

        texture_type = self.texture_resource_type(args[0])
        if self.storage_image_texture_operation_expression(func_name, texture_type):
            return False

        if is_texture_sample_operation(func_name):
            return not self.is_multisample_texture_resource(texture_type)

        if is_texture_query_lod_operation(func_name):
            return not (
                self.is_multisample_texture_resource(texture_type)
                or self.is_storage_image_resource(texture_type)
            )

        if is_texture_sample_offset_operation(func_name):
            return self.texture_sample_supports_offset(texture_type)

        if is_projected_texture_operation(func_name):
            texture_type = self.resource_base_type(texture_type)
            if (
                is_projected_texture_basic_offset_operation(func_name)
                or is_projected_texture_lod_offset_operation(func_name)
                or is_projected_texture_grad_offset_operation(func_name)
            ):
                return texture_type.startswith("texture2d<") or texture_type.startswith(
                    "texture2d_array<"
                )
            return (
                texture_type.startswith("texture1d<")
                or texture_type.startswith("texture2d<")
                or texture_type.startswith("texture2d_array<")
                or texture_type.startswith("texture3d<")
                or texture_type.startswith("texturecube<")
            )

        return True

    def global_texture_names(self):
        return {
            texture_variable.name
            for texture_variable, _, _, _ in self.texture_variables
            if getattr(texture_variable, "name", None)
        }

    def global_acceleration_structure_names(self):
        return {
            acceleration_structure_variable.name
            for acceleration_structure_variable, _, _, _ in (
                self.acceleration_structure_variables
            )
            if getattr(acceleration_structure_variable, "name", None)
        }

    def global_visible_function_table_names(self):
        return {
            visible_function_table_variable.name
            for visible_function_table_variable, _, _, _ in (
                self.visible_function_table_variables
            )
            if getattr(visible_function_table_variable, "name", None)
        }

    def global_intersection_function_table_names(self):
        return {
            intersection_function_table_variable.name
            for intersection_function_table_variable, _, _, _ in (
                self.intersection_function_table_variables
            )
            if getattr(intersection_function_table_variable, "name", None)
        }

    def global_structured_buffer_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _ in self.structured_buffer_variables
            if getattr(buffer_variable, "name", None)
        }

    def global_metal_buffer_resource_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _, _ in self.metal_buffer_resource_variables
            if getattr(buffer_variable, "name", None)
        }

    def structured_buffer_requires_length(self, name):
        return bool(name and name in self.global_structured_buffer_length_dependencies)

    def structured_buffer_parameter_requires_length(self, func_name, name):
        return bool(
            func_name
            and name
            and name
            in self.function_structured_buffer_length_dependencies.get(func_name, set())
        )

    def global_structured_buffer_requires_counter(self, name):
        if not name:
            return False
        for buffer_variable, _, buffer_type, _ in self.structured_buffer_variables:
            if getattr(buffer_variable, "name", None) == name:
                return self.structured_buffer_requires_counter(buffer_type)
        return False

    def global_glsl_buffer_block_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _ in self.glsl_buffer_block_variables
            if getattr(buffer_variable, "name", None)
        }

    def global_sampler_names(self):
        return {
            sampler_variable.name
            for sampler_variable, _, _ in self.sampler_variables
            if getattr(sampler_variable, "name", None)
        }

    def required_function_textures(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (texture_variable, texture_type, array_size)
            for texture_variable, _, texture_type, array_size in self.texture_variables
            if getattr(texture_variable, "name", None) in dependencies
        ]

    def required_function_acceleration_structures(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (
                acceleration_structure_variable,
                acceleration_structure_type,
                array_size,
            )
            for (
                acceleration_structure_variable,
                _,
                acceleration_structure_type,
                array_size,
            ) in self.acceleration_structure_variables
            if getattr(acceleration_structure_variable, "name", None) in dependencies
        ]

    def required_function_visible_function_tables(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (
                visible_function_table_variable,
                visible_function_table_type,
                array_size,
            )
            for (
                visible_function_table_variable,
                _,
                visible_function_table_type,
                array_size,
            ) in self.visible_function_table_variables
            if getattr(visible_function_table_variable, "name", None) in dependencies
        ]

    def required_function_intersection_function_tables(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (
                intersection_function_table_variable,
                intersection_function_table_type,
                array_size,
            )
            for (
                intersection_function_table_variable,
                _,
                intersection_function_table_type,
                array_size,
            ) in self.intersection_function_table_variables
            if getattr(intersection_function_table_variable, "name", None)
            in dependencies
        ]

    def required_function_samplers(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (sampler_variable, array_size)
            for sampler_variable, _, array_size in self.sampler_variables
            if getattr(sampler_variable, "name", None) in dependencies
        ]

    def required_function_structured_buffers(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, buffer_type, array_size)
            for (
                buffer_variable,
                _,
                buffer_type,
                array_size,
            ) in self.structured_buffer_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    def required_function_metal_buffer_resources(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, buffer_type, array_size, address_space)
            for (
                buffer_variable,
                _,
                buffer_type,
                array_size,
                address_space,
            ) in self.metal_buffer_resource_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    def required_function_glsl_buffer_blocks(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, block, array_size)
            for (
                buffer_variable,
                _,
                block,
                array_size,
            ) in self.glsl_buffer_block_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    def generate_function_call_arguments(self, func_name, call_args):
        call_args = list(call_args or [])
        parameter_infos = self.function_parameter_infos.get(func_name, [])
        if len(call_args) != len(parameter_infos):
            parameter_infos = []
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        args = []
        for index, arg in enumerate(call_args):
            if index in skipped_indices:
                continue
            param_name, param_type = (
                parameter_infos[index] if index < len(parameter_infos) else (None, None)
            )
            if self.is_unsupported_metal_ray_function_table_array_parameter(
                func_name, index
            ):
                continue
            if self.is_unsupported_metal_acceleration_structure_array_parameter(
                func_name, index
            ):
                continue
            args.append(self.generate_expression(arg))
            if self.structured_buffer_parameter_requires_length(func_name, param_name):
                length = self.structured_buffer_length_data_argument(arg)
                if length is not None:
                    args.append(length)
            if param_type is not None and self.structured_buffer_requires_counter(
                param_type
            ):
                counter = self.structured_buffer_counter_data_argument(arg)
                if counter is not None:
                    args.append(counter)
        return args

    def is_unsupported_metal_ray_function_table_array_parameter(self, func_name, index):
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        if index >= len(parameter_nodes):
            return False
        parameter = parameter_nodes[index]
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        previous_function_name = self.current_function_name
        self.current_function_name = func_name
        try:
            return (
                self.metal_ray_function_table_array_parameter_kind(raw_type, parameter)
                is not None
            )
        finally:
            self.current_function_name = previous_function_name

    def is_unsupported_metal_acceleration_structure_array_parameter(
        self, func_name, index
    ):
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        if index >= len(parameter_nodes):
            return False
        parameter = parameter_nodes[index]
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        previous_function_name = self.current_function_name
        self.current_function_name = func_name
        try:
            return (
                self.metal_acceleration_structure_array_parameter_kind(
                    raw_type, parameter
                )
                is not None
            )
        finally:
            self.current_function_name = previous_function_name

    def required_function_resource_argument_names(self, func_name):
        names = [
            texture_variable.name
            for texture_variable, _, _ in self.required_function_textures(func_name)
        ]
        names.extend(
            acceleration_structure_variable.name
            for acceleration_structure_variable, _, _ in (
                self.required_function_acceleration_structures(func_name)
            )
        )
        names.extend(
            visible_function_table_variable.name
            for visible_function_table_variable, _, _ in (
                self.required_function_visible_function_tables(func_name)
            )
        )
        names.extend(
            intersection_function_table_variable.name
            for intersection_function_table_variable, _, _ in (
                self.required_function_intersection_function_tables(func_name)
            )
        )
        for (
            buffer_variable,
            buffer_type,
            _array_size,
            _address_space,
        ) in self.required_function_metal_buffer_resources(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                names.append(buffer_name)
        for (
            buffer_variable,
            buffer_type,
            _array_size,
        ) in self.required_function_structured_buffers(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if not buffer_name:
                continue
            names.append(buffer_name)
            if self.structured_buffer_requires_length(buffer_name):
                names.append(self.structured_buffer_length_parameter_name(buffer_name))
            if self.structured_buffer_requires_counter(buffer_type):
                names.append(self.structured_buffer_counter_parameter_name(buffer_name))
        names.extend(
            buffer_variable.name
            for buffer_variable, _, _ in self.required_function_glsl_buffer_blocks(
                func_name
            )
        )
        names.extend(
            sampler_variable.name
            for sampler_variable, _ in self.required_function_samplers(func_name)
        )
        return names

    def iter_ast_nodes(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (list, tuple, set)):
            for item in node:
                yield from self.iter_ast_nodes(item)
            return
        if not hasattr(node, "__dict__"):
            return
        yield node
        for key, value in vars(node).items():
            if key in {"parent", "annotations"}:
                continue
            yield from self.iter_ast_nodes(value)

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def initial_literal_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)
        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def visible_literal_int_constants(self, func):
        visible_constants = self.initial_literal_int_constants(func)

        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                name = getattr(node, "name", None)
                if not name:
                    continue

                visible_constants.pop(name, None)
                if "const" not in getattr(node, "qualifiers", []):
                    continue

                value = self.literal_int_value(
                    getattr(node, "initial_value", None), visible_constants
                )
                if value is not None:
                    visible_constants[name] = value

        return visible_constants

    def function_call_name(self, call):
        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if isinstance(func_expr, str):
            return func_expr
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            return func_expr.name
        return None

    def member_access_is_image_load_component(self, expr):
        if not isinstance(getattr(expr, "object", None), FunctionCallNode):
            return False
        if self.function_call_name(expr.object) != "imageLoad":
            return False
        member = str(getattr(expr, "member", ""))
        return bool(member) and all(char in "xyzwrgba" for char in member)

    def supported_image_formats(self):
        return supported_image_formats()

    def scalar_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) == 1
        }

    def vector_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) in {2, 4}
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        name = getattr(value, "name", None)
        if name is not None:
            return str(name)
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        return str(value)

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "group",
            "packoffset",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def explicit_resource_binding_index(
        self, node, attribute_names=(), register_prefixes=()
    ):
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in attribute_names:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], register_prefixes)
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_resource_set_index(self, node):
        if not hasattr(node, "attributes"):
            return None

        selected_set = None
        selected_description = None
        node_name = self.resource_node_name(node, "<anonymous>")
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in {"set", "space"}:
                source = (
                    self.attribute_value_to_string(arguments[0]) if arguments else None
                )
                set_index = (
                    self.binding_index_value(arguments[0], ("set", "space"))
                    if len(arguments) == 1
                    else None
                )
                description = (
                    f"{attr_name} {source if source is not None else '<missing>'}"
                )
            elif attr_name == "register" and len(arguments) > 1:
                source = self.attribute_value_to_string(arguments[1])
                set_index = self.binding_index_value(arguments[1], ("space", "set"))
                description = (
                    f"register space {source if source is not None else '<missing>'}"
                )
            else:
                continue

            if set_index is None:
                raise ValueError(
                    "Invalid Metal resource binding metadata for "
                    f"'{node_name}': {description} must resolve to a concrete "
                    "integer set"
                )
            if selected_set is None:
                selected_set = set_index
                selected_description = description
                continue
            if selected_set != set_index:
                raise ValueError(
                    "Conflicting Metal resource binding metadata for "
                    f"'{node_name}': {selected_description} differs from "
                    f"{description}"
                )
        return selected_set

    def semantic_from_node(self, node):
        if hasattr(node, "semantic"):
            return node.semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_metal_texture_element_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
                or self.is_metal_address_space_attribute(attr)
                or self.is_metal_struct_member_abi_attribute(attr)
                or self.metal_interpolation_attribute_name(attr) is not None
                or self.is_precision_qualifier_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def is_precision_qualifier_attribute(self, attr):
        return str(getattr(attr, "name", attr)).lower() in {"lowp", "mediump", "highp"}

    def is_metal_address_space_attribute(self, attr):
        name = str(getattr(attr, "name", "")).lower()
        return name in {
            "constant",
            "device",
            "global",
            "groupshared",
            "local",
            "object_data",
            "objectdata",
            "private",
            "ray_data",
            "raydata",
            "shared",
            "storage",
            "thread",
            "threadgroup",
            "workgroup",
        }

    def metal_stage_control_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]

        valid_names = {
            "cw",
            "ccw",
            "domain",
            "equal_spacing",
            "fractional_even_spacing",
            "fractional_odd_spacing",
            "invocations",
            "isolines",
            "layout",
            "line_strip",
            "lines",
            "lines_adjacency",
            "local_size_x",
            "local_size_y",
            "local_size_z",
            "max_primitives",
            "maxtessfactor",
            "max_total_threads_per_threadgroup",
            "max_vertices",
            "maxvertexcount",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
            "point_mode",
            "points",
            "quads",
            "threadgroup_size",
            "threads_per_threadgroup",
            "triangles",
            "triangles_adjacency",
            "triangle_strip",
            "vertices",
        }
        if normalized in valid_names:
            return normalized
        return None

    def function_return_semantic(self, func):
        semantic = getattr(func, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(func, "attributes"):
            return None
        for attr in func.attributes:
            if self.metal_stage_control_attribute_name(attr):
                continue
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_metal_texture_element_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
                or self.is_precision_qualifier_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def map_non_stage_function_semantic(self, semantic):
        if self.is_legacy_glsl_fragment_output_semantic(semantic):
            return ""
        return self.map_semantic(semantic)

    def is_legacy_glsl_fragment_output_semantic(self, semantic):
        if semantic is None:
            return False
        semantic_text = str(semantic)
        return (
            semantic_text == "gl_FragData"
            or semantic_text.startswith("gl_FragData[")
            or semantic_text == "gl_FragColor"
            or re.fullmatch(r"gl_FragColor\d+", semantic_text) is not None
        )

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
            return self.map_type(vtype)

        if isinstance(vtype, (PointerType, ReferenceType)):
            return self.map_type(vtype)

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_image_base_type_with_format(base_type, node)
            return f"{base_mapped}{array_suffix}"

        return self.map_image_base_type_with_format(vtype_str, node)

    def map_image_base_type_with_format(self, vtype, node=None):
        base_type = self.resource_base_type(vtype)
        explicit_format = (
            explicit_image_format(node, self.attribute_value_to_string)
            if node is not None
            else None
        )
        component_type = self.metal_texture_element_attribute_type(node) or (
            self.scalar_image_format_components().get(explicit_format)
            or self.vector_image_format_components().get(explicit_format)
        )
        texture_types = {
            "image1D": "texture1d",
            "iimage1D": "texture1d",
            "uimage1D": "texture1d",
            "image2D": "texture2d",
            "iimage2D": "texture2d",
            "uimage2D": "texture2d",
            "image3D": "texture3d",
            "iimage3D": "texture3d",
            "uimage3D": "texture3d",
            "image1DArray": "texture1d_array",
            "iimage1DArray": "texture1d_array",
            "uimage1DArray": "texture1d_array",
            "image2DArray": "texture2d_array",
            "iimage2DArray": "texture2d_array",
            "uimage2DArray": "texture2d_array",
            "image2DMS": "texture2d_ms",
            "iimage2DMS": "texture2d_ms",
            "uimage2DMS": "texture2d_ms",
            "image2DMSArray": "texture2d_ms_array",
            "iimage2DMSArray": "texture2d_ms_array",
            "uimage2DMSArray": "texture2d_ms_array",
            "imageCube": "texture2d_array",
        }
        texture_type = texture_types.get(base_type)
        if texture_type:
            if component_type is None:
                component_type = self.default_image_component_type(base_type)
            access = (
                explicit_image_access(node, self.attribute_value_to_string)
                or "read_write"
            )
            if texture_type in {"texture2d_ms", "texture2d_ms_array"}:
                access = "read"
            return f"{texture_type}<{component_type}, access::{access}>"
        return self.map_type(vtype)

    def is_metal_texture_element_attribute(self, attr):
        return str(getattr(attr, "name", "")).lower() in {
            "metal_texture_element_half",
        }

    def metal_texture_element_attribute_type(self, node):
        if node is None:
            return None
        for attr in getattr(node, "attributes", []) or []:
            if self.is_metal_texture_element_attribute(attr):
                return "half"
        return None

    def default_image_component_type(self, vtype):
        base_type = self.resource_base_type(vtype)
        if base_type.startswith("iimage"):
            return "int"
        if base_type.startswith("uimage"):
            return "uint"
        return "float"

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "globallycoherent",
            "restrict",
            "volatile",
        }

    def resource_base_type(self, vtype):
        if vtype is None:
            return ""
        if isinstance(vtype, PointerType):
            return self.resource_base_type(vtype.pointee_type)
        if isinstance(vtype, ReferenceType):
            return self.resource_base_type(vtype.referenced_type)
        if self.is_array_type_node(vtype):
            return self.resource_base_type(vtype.element_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype = self.convert_type_node_to_string(vtype)
        vtype = str(vtype)
        if "[" in vtype and "]" in vtype:
            base_type, _ = parse_array_type(vtype)
            return base_type
        return vtype

    def glsl_buffer_block_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == "glsl_buffer_block":
                return attr
        return None

    def glsl_buffer_block_layout(self, node):
        attr = self.glsl_buffer_block_attribute(node)
        arguments = getattr(attr, "arguments", []) if attr is not None else []
        if arguments:
            layout = self.attribute_value_to_string(arguments[0])
            if layout:
                return layout
        return "std430"

    def is_glsl_buffer_block_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        return bool(attr_name and str(attr_name).lower() == "glsl_buffer_block")

    def is_glsl_buffer_block_variable(self, node, vtype=None):
        if self.glsl_buffer_block_attribute(node) is None:
            return False
        type_name = self.resource_base_type(vtype or glsl_buffer_block_node_type(node))
        return str(type_name) in self.structs_by_name

    def collect_glsl_buffer_block_struct_names(self, global_vars):
        names = set()
        for node in global_vars:
            node_type = glsl_buffer_block_node_type(node)
            if not self.is_glsl_buffer_block_variable(node, node_type):
                continue
            type_name = self.resource_base_type(node_type)
            names.add(str(type_name))
        return names

    def collect_lowered_glsl_buffer_block_parameters(self, parameters):
        return collect_lowered_glsl_buffer_blocks(
            parameters,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: self.literal_int_value(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="metal_type",
            unsupported_type_message=(
                "type is not supported by Metal pointer/offset lowering"
            ),
        )

    def collect_unsupported_glsl_buffer_block_parameter_names(self, parameters):
        names = set()
        for param in parameters or []:
            param_name = getattr(param, "name", None)
            param_type = glsl_buffer_block_node_type(param)
            if (
                param_name
                and param_name not in self.current_glsl_buffer_block_parameters
                and self.is_glsl_buffer_block_variable(param, param_type)
            ):
                names.add(param_name)
        return names

    def collect_unsupported_glsl_buffer_block_functions(self, functions):
        unsupported = {}
        omitted_struct_names = set(self.glsl_buffer_block_struct_names)
        if not omitted_struct_names:
            return unsupported

        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            return_type = self.type_name_string(getattr(func, "return_type", "void"))
            return_base = str(self.resource_base_type(return_type))
            if return_base in omitted_struct_names:
                unsupported[func_name] = {
                    "return_type": return_type,
                    "reason": (
                        f"return type references omitted GLSL buffer block struct {return_base}"
                    ),
                }
                continue

            param_list = getattr(func, "parameters", getattr(func, "params", []))
            lowered_params, _, _ = self.collect_lowered_glsl_buffer_block_parameters(
                param_list
            )
            for param in param_list:
                param_name = getattr(param, "name", None)
                param_type = getattr(
                    param,
                    "param_type",
                    getattr(param, "var_type", getattr(param, "vtype", None)),
                )
                param_base = str(self.resource_base_type(param_type))
                if (
                    param_base in omitted_struct_names
                    and param_name not in lowered_params
                ):
                    unsupported[func_name] = {
                        "return_type": return_type,
                        "reason": (
                            f"parameter {param_name} references omitted GLSL "
                            f"buffer block struct {param_base}"
                        ),
                    }
                    break
        return unsupported

    def glsl_buffer_block_lowering_failure_detail(self, type_name, var_name=None):
        if var_name:
            reason = self.current_glsl_buffer_block_parameter_failures.get(var_name)
            if reason:
                return reason
            reason = self.glsl_buffer_block_lowering_failures.get(var_name)
            if reason:
                return reason
        type_name = str(self.resource_base_type(type_name))
        return self.current_glsl_buffer_block_parameter_struct_failures.get(
            type_name
        ) or self.glsl_buffer_block_struct_lowering_failures.get(type_name)

    def glsl_buffer_block_parameter_diagnostics(self, target, parameters, indent=0):
        code = ""
        indent_str = "  " * indent
        for param in parameters or []:
            param_name = getattr(param, "name", None)
            param_type = glsl_buffer_block_node_type(param)
            if (
                not param_name
                or param_name in self.current_glsl_buffer_block_parameters
                or param_name
                not in self.current_unsupported_glsl_buffer_block_parameters
                or not self.is_glsl_buffer_block_variable(param, param_type)
            ):
                continue
            code += indent_str + self.glsl_buffer_block_diagnostic(
                target, param_type, param_name, param, declaration_kind="parameter"
            )
        return code

    def glsl_buffer_block_member_access(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        var_name = self.expression_name(object_expr)
        member_name = getattr(expr, "member", None)
        if var_name:
            block = self.current_glsl_buffer_block_parameters.get(
                var_name
            ) or self.lowered_glsl_buffer_blocks.get(var_name)
            if block is not None:
                buffer_expr = self.generate_expression(object_expr)
                member = block["members"].get(member_name)
                if member is None:
                    return None
                return {
                    "buffer": buffer_expr,
                    "member": member_name,
                    "readonly": block["readonly"],
                    **member,
                }

        parent = self.glsl_buffer_block_array_access(
            object_expr
        ) or self.glsl_buffer_block_member_access(object_expr)
        if parent is None or not parent.get("members"):
            return None
        member = parent["members"].get(member_name)
        if member is None:
            return None
        return {
            "buffer": parent["buffer"],
            "member": f"{parent['member']}.{member_name}",
            "readonly": parent["readonly"],
            **member,
            "offset": byte_offset_add(parent["offset"], member["offset"]),
        }

    def glsl_buffer_block_array_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        member = self.glsl_buffer_block_member_access(array_expr)
        if member is None or not member.get("is_array"):
            return None
        index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
        index = self.generate_expression(index_expr)
        offset = byte_offset_expression(member["offset"], index, member["stride"])
        return {**member, "offset": offset, "offset_expr": offset}

    def metal_scalar_load(self, component_type, buffer_name, offset):
        if component_type == "bool":
            return (
                f"((*reinterpret_cast<const device uint*>"
                f"({buffer_name} + {offset})) != 0u)"
            )
        return (
            f"(*reinterpret_cast<const device {component_type}*>"
            f"({buffer_name} + {offset}))"
        )

    def metal_scalar_store(self, component_type, buffer_name, offset, value):
        if component_type == "bool":
            return (
                f"(*reinterpret_cast<device uint*>"
                f"({buffer_name} + {offset})) = (({value}) ? 1u : 0u)"
            )
        return (
            f"(*reinterpret_cast<device {component_type}*>"
            f"({buffer_name} + {offset})) = {value}"
        )

    def metal_buffer_load(self, buffer_name, offset, access):
        if access.get("matrix_columns"):
            columns = []
            for _, column_offset in matrix_column_offsets(
                offset, access["matrix_columns"], access["column_stride"]
            ):
                column_access = {
                    "components": access["matrix_rows"],
                    "component_type": "float",
                    "metal_type": f"float{access['matrix_rows']}",
                }
                columns.append(
                    self.metal_buffer_load(buffer_name, column_offset, column_access)
                )
            return f"{access['metal_type']}({', '.join(columns)})"

        if access["components"] == 1:
            return self.metal_scalar_load(access["component_type"], buffer_name, offset)

        values = []
        for _, component_offset in vector_component_offsets(
            offset, access["components"]
        ):
            values.append(
                self.metal_scalar_load(
                    access["component_type"], buffer_name, component_offset
                )
            )
        return f"{access['metal_type']}({', '.join(values)})"

    def next_metal_temp_variable(self, prefix):
        name = f"__crossgl_{prefix}_{self.metal_temp_variable_index}"
        self.metal_temp_variable_index += 1
        return name

    def metal_buffer_store(self, buffer_name, offset, value, access):
        if access.get("matrix_columns"):
            return self.metal_matrix_store(buffer_name, offset, value, access)

        if access["components"] == 1:
            if access.get("layout_type") != access.get("type"):
                value = f"{access['component_type']}({value})"
            return self.metal_scalar_store(
                access["component_type"], buffer_name, offset, value
            )

        temp_name = self.next_metal_temp_variable("buffer_store")
        lines = [f"{access['metal_type']} {temp_name} = {value}"]
        for component, component_offset in vector_component_offsets(
            offset, access["components"]
        ):
            field = "xyzw"[component]
            lines.append(
                self.metal_scalar_store(
                    access["component_type"],
                    buffer_name,
                    component_offset,
                    f"{temp_name}.{field}",
                )
            )
        return "\n".join(lines)

    def metal_matrix_store(self, buffer_name, offset, value, access):
        temp_name = self.next_metal_temp_variable("matrix_store")
        lines = [f"{access['metal_type']} {temp_name} = {value}"]
        for column, column_offset in matrix_column_offsets(
            offset, access["matrix_columns"], access["column_stride"]
        ):
            for row, element_offset in vector_component_offsets(
                column_offset, access["matrix_rows"]
            ):
                field = "xyzw"[row]
                lines.append(
                    self.metal_scalar_store(
                        "float",
                        buffer_name,
                        element_offset,
                        f"{temp_name}[{column}].{field}",
                    )
                )
        return "\n".join(lines)

    def metal_aggregate_has_array_member(self, access):
        for member in access["members"].values():
            if member.get("is_array"):
                return True
            if member.get("members") and self.metal_aggregate_has_array_member(member):
                return True
        return False

    def metal_aggregate_helper_suffix(self, access):
        return "".join(
            char if char.isalnum() or char == "_" else "_"
            for char in access["metal_type"]
        )

    def metal_aggregate_layout_signature(self, access):
        parts = []

        def visit(member_name, member):
            fields = [
                member_name,
                str(member.get("type")),
                str(member.get("layout_type")),
                str(member.get("offset")),
                str(member.get("size")),
                str(member.get("align")),
                str(member.get("components")),
                str(member.get("component_type")),
                str(member.get("matrix_columns")),
                str(member.get("matrix_rows")),
                str(member.get("column_stride")),
                str(member.get("is_array")),
                str(member.get("array_count")),
                str(member.get("stride")),
                str(member.get("runtime_array")),
            ]
            parts.append(":".join(fields))
            for child_name, child in (member.get("members") or {}).items():
                visit(f"{member_name}.{child_name}", child)

        for field_name, member in access["members"].items():
            visit(field_name, member)
        return sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]

    def metal_aggregate_load_helper_name(self, access):
        helper_name = (
            f"__crossgl_load_glsl_buffer_"
            f"{self.metal_aggregate_helper_suffix(access)}_"
            f"{self.metal_aggregate_layout_signature(access)}"
        )
        self.required_glsl_buffer_aggregate_load_helpers[helper_name] = access
        return helper_name

    def metal_aggregate_load_assignments(
        self, target_name, buffer_name, offset, access, indent=1
    ):
        indent_str = "    " * indent
        lines = []
        for field_name, member in access["members"].items():
            member_offset = byte_offset_add(offset, member["offset"])
            member_target = f"{target_name}.{field_name}"
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access["readonly"],
            }
            if member.get("is_array"):
                array_count = member.get("array_count")
                if member.get("runtime_array") or array_count is None:
                    return None
                for index in range(array_count):
                    element_offset = byte_offset_add(
                        member_offset, index * member["stride"]
                    )
                    element_target = f"{member_target}[{index}]"
                    if member.get("members"):
                        nested_lines = self.metal_aggregate_load_assignments(
                            element_target,
                            buffer_name,
                            element_offset,
                            field_access,
                            indent,
                        )
                        if nested_lines is None:
                            return None
                        lines.extend(nested_lines)
                    else:
                        value = self.metal_buffer_load(
                            buffer_name, element_offset, field_access
                        )
                        lines.append(f"{indent_str}{element_target} = {value};")
                continue
            if member.get("members"):
                nested_lines = self.metal_aggregate_load_assignments(
                    member_target, buffer_name, member_offset, field_access, indent
                )
                if nested_lines is None:
                    return None
                lines.extend(nested_lines)
            else:
                value = self.metal_buffer_load(buffer_name, member_offset, field_access)
                lines.append(f"{indent_str}{member_target} = {value};")
        return lines

    def generate_glsl_buffer_aggregate_load_helpers(self):
        if not self.required_glsl_buffer_aggregate_load_helpers:
            return ""

        helpers = []
        for helper_name, access in sorted(
            self.required_glsl_buffer_aggregate_load_helpers.items()
        ):
            lines = [
                f"{access['metal_type']} {helper_name}(const device uchar* buffer, uint offset) {{",
                f"    {access['metal_type']} result;",
            ]
            assignments = self.metal_aggregate_load_assignments(
                "result", "buffer", "offset", access
            )
            if assignments is None:
                continue
            lines.extend(assignments)
            lines.extend(["    return result;", "}"])
            helpers.append("\n".join(lines) + "\n\n")
        return "".join(helpers)

    def metal_aggregate_load(self, buffer_name, offset, access):
        if self.metal_aggregate_has_array_member(access):
            helper_name = self.metal_aggregate_load_helper_name(access)
            return f"{helper_name}({buffer_name}, {offset})"

        values = []
        for field_name, member in access["members"].items():
            if member.get("is_array"):
                return None
            member_offset = byte_offset_add(offset, member["offset"])
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access["readonly"],
            }
            if member.get("members"):
                value = self.metal_aggregate_load(
                    buffer_name, member_offset, field_access
                )
            else:
                value = self.metal_buffer_load(buffer_name, member_offset, field_access)
            if value is None:
                return None
            values.append(value)
        return format_struct_constructor_expression(self, access["metal_type"], values)

    def metal_aggregate_store_members(self, buffer_name, offset, value, access):
        lines = []
        for field_name, member in access["members"].items():
            member_offset = byte_offset_add(offset, member["offset"])
            member_value = f"{value}.{field_name}"
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access["readonly"],
            }
            if member.get("is_array"):
                array_count = member.get("array_count")
                if member.get("runtime_array") or array_count is None:
                    return None
                for index in range(array_count):
                    element_offset = byte_offset_add(
                        member_offset, index * member["stride"]
                    )
                    element_value = f"{member_value}[{index}]"
                    if member.get("members"):
                        nested_stores = self.metal_aggregate_store_members(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if nested_stores is None:
                            return None
                        lines.extend(nested_stores)
                    else:
                        store = self.metal_buffer_store(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if store is None:
                            return None
                        lines.extend(store.splitlines())
                continue
            if member.get("members"):
                nested_stores = self.metal_aggregate_store_members(
                    buffer_name, member_offset, member_value, field_access
                )
                if nested_stores is None:
                    return None
                lines.extend(nested_stores)
            else:
                store = self.metal_buffer_store(
                    buffer_name, member_offset, member_value, field_access
                )
                if store is None:
                    return None
                lines.extend(store.splitlines())
        return lines

    def metal_aggregate_store(self, buffer_name, offset, value, access):
        temp_name = self.next_metal_temp_variable("aggregate_store")
        stores = self.metal_aggregate_store_members(
            buffer_name, offset, temp_name, access
        )
        if stores is None:
            return (
                "/* unsupported Metal GLSL buffer block aggregate store: "
                "array fields require element-wise stores */"
            )
        return "\n".join([f"{access['metal_type']} {temp_name} = {value}", *stores])

    def metal_matrix_compound_store(self, buffer_name, offset, value, op, access):
        compound_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
        }
        binary_op = compound_ops.get(op)
        if binary_op is None:
            return (
                "/* unsupported Metal GLSL buffer block matrix compound store: "
                "requires explicit matrix operation lowering */"
            )

        current = self.metal_buffer_load(buffer_name, offset, access)
        rhs = f"({current} {binary_op} {value})"
        return self.metal_matrix_store(buffer_name, offset, rhs, access)

    def metal_buffer_compound_store_diagnostic(self, op, access):
        return (
            "/* unsupported Metal GLSL buffer block compound store: "
            f"operator {op} is not supported for "
            f"{access['component_type']} buffer members */"
        )

    def generate_glsl_buffer_block_member_load(self, expr):
        access = self.glsl_buffer_block_member_access(expr)
        if access is None or access.get("is_array"):
            return None
        if access.get("members"):
            return self.metal_aggregate_load(access["buffer"], access["offset"], access)
        return self.metal_buffer_load(access["buffer"], access["offset"], access)

    def generate_glsl_buffer_block_array_load(self, expr):
        access = self.glsl_buffer_block_array_access(expr)
        if access is None:
            return None
        if access.get("members"):
            return self.metal_aggregate_load(
                access["buffer"], access["offset_expr"], access
            )
        return self.metal_buffer_load(access["buffer"], access["offset_expr"], access)

    def generate_glsl_buffer_block_store(self, target, rhs, op):
        access = self.glsl_buffer_block_array_access(target)
        if access is None:
            access = self.glsl_buffer_block_member_access(target)
            if access is None or access.get("is_array"):
                return None
            offset = access["offset"]
        else:
            offset = access["offset_expr"]

        if access.get("readonly"):
            return (
                "/* unsupported Metal GLSL buffer block store: "
                "readonly device buffer cannot be written */"
            )
        if access.get("members"):
            if op != "=":
                return (
                    "/* unsupported Metal GLSL buffer block aggregate compound "
                    "store: assign a full aggregate value explicitly */"
                )
            return self.metal_aggregate_store(access["buffer"], offset, rhs, access)

        if access.get("matrix_columns"):
            if op != "=":
                return self.metal_matrix_compound_store(
                    access["buffer"], offset, rhs, op, access
                )
            return self.metal_matrix_store(access["buffer"], offset, rhs, access)

        if op != "=":
            binary_op = glsl_buffer_compound_binary_operator(
                op, access["component_type"]
            )
            if binary_op is None:
                return self.metal_buffer_compound_store_diagnostic(op, access)
            current = self.metal_buffer_load(access["buffer"], offset, access)
            rhs = f"({current} {binary_op} {rhs})"

        return self.metal_buffer_store(access["buffer"], offset, rhs, access)

    def buffer_atomic_operations(self):
        return {
            "atomicAdd": ("fetch_add", 2),
            "atomicMin": ("fetch_min", 2),
            "atomicMax": ("fetch_max", 2),
            "atomicAnd": ("fetch_and", 2),
            "atomicOr": ("fetch_or", 2),
            "atomicXor": ("fetch_xor", 2),
            "atomicExchange": ("exchange", 2),
            "atomicCompSwap": ("compare_exchange", 3),
        }

    def glsl_buffer_block_atomic_access(self, target):
        access = self.glsl_buffer_block_array_access(target)
        if access is not None:
            return access, access["offset_expr"]
        access = self.glsl_buffer_block_member_access(target)
        if access is None or access.get("is_array"):
            return None, None
        return access, access["offset"]

    def unsupported_glsl_buffer_block_atomic_call(
        self, target, operation, reason, access=None
    ):
        result_type = self.expression_result_type(target) or "uint"
        component_type = access.get("component_type") if access else None
        if component_type is not None:
            zero_value = "0u" if component_type == "uint" else "0"
        else:
            zero_value = "0u" if self.type_name_string(result_type) == "uint" else "0"
        return (
            "/* unsupported Metal GLSL buffer block atomic: "
            f"{operation} {reason} */ {zero_value}"
        )

    def generate_glsl_buffer_block_atomic_call(self, func_name, args):
        operations = self.buffer_atomic_operations()
        operation_info = operations.get(func_name)
        if operation_info is None or not args:
            return None

        operation, expected_args = operation_info
        if len(args) < expected_args:
            return None

        target = args[0]
        access, offset = self.glsl_buffer_block_atomic_access(target)
        if access is None:
            return None
        if access.get("readonly"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "cannot write readonly device buffer",
                access,
            )
        if access.get("components") != 1 or access.get("matrix_columns"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "requires a scalar int or uint buffer member",
                access,
            )
        if access.get("component_type") not in {"int", "uint"}:
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "currently supports only int or uint buffer members",
                access,
            )

        if operation == "compare_exchange":
            self.required_buffer_atomic_compare_helpers.add(access["component_type"])
            helper_name = self.buffer_atomic_compare_helper_name(
                access["component_type"]
            )
            compare_value = self.generate_expression_with_expected(
                args[1], access["type"]
            )
            replacement = self.generate_expression_with_expected(
                args[2], access["type"]
            )
            return (
                f"{helper_name}({access['buffer']}, {offset}, "
                f"{compare_value}, {replacement})"
            )

        atomic_type = f"atomic_{access['component_type']}"
        atomic_target = (
            f"reinterpret_cast<device {atomic_type}*>({access['buffer']} + {offset})"
        )
        value = self.generate_expression_with_expected(args[1], access["type"])
        return f"atomic_{operation}_explicit({atomic_target}, {value}, memory_order_relaxed)"

    def glsl_buffer_block_diagnostic(
        self, target, type_name, var_name=None, node=None, declaration_kind=None
    ):
        declaration = str(self.resource_base_type(type_name))
        if declaration_kind:
            declaration = f"{declaration_kind} {declaration}"
        if var_name:
            declaration += f" {var_name}"
        details = ""
        if node is not None:
            layout = self.glsl_buffer_block_layout(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            details = f" ({layout}"
            if binding is not None:
                details += f", binding = {binding}"
            details += ")"
        failure_detail = self.glsl_buffer_block_lowering_failure_detail(
            type_name, var_name
        )
        if failure_detail:
            details += f"; {failure_detail}"
        return (
            f"// unsupported {target} GLSL buffer block {declaration}{details}: "
            "mixed metadata/runtime-array layout requires explicit pointer/offset "
            "lowering\n"
        )

    def unsupported_glsl_buffer_block_struct_placeholder(self, target, type_name):
        type_name = str(self.resource_base_type(type_name))
        return (
            f"// unsupported {target} GLSL buffer block struct {type_name} "
            "omitted: no target-side fallback declaration emitted\n"
        )

    def unsupported_glsl_buffer_block_variable_placeholder(
        self, target, type_name, var_name
    ):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        return (
            f"// unsupported {target} GLSL buffer block variable {declaration} "
            "omitted: no target-side fallback declaration emitted\n"
        )

    def unsupported_glsl_buffer_block_function_placeholder(
        self, target, func_name, info
    ):
        reason = info.get("reason", "signature references omitted GLSL buffer block")
        return (
            f"// unsupported {target} GLSL buffer block function {func_name} "
            f"omitted: {reason}\n"
        )

    def unsupported_glsl_buffer_block_function_call(self, func_name):
        if (
            not func_name
            or func_name not in self.unsupported_glsl_buffer_block_functions
        ):
            return None
        info = self.unsupported_glsl_buffer_block_functions[func_name]
        return_type = info.get("return_type") or self.current_expression_expected_type
        diagnostic = (
            f"unsupported Metal GLSL buffer block function call {func_name}: "
            "target function omitted"
        )
        if self.map_type(return_type) == "void":
            return f"/* {diagnostic} */"
        fallback = self.diagnostic_zero_value_for_type(return_type)
        return f"{fallback} /* {diagnostic} */"

    def is_unsupported_glsl_buffer_block_struct_type(self, vtype):
        return str(self.resource_base_type(vtype)) in (
            self.unsupported_glsl_buffer_block_struct_names
        )

    def unsupported_glsl_buffer_block_local_variable_placeholder(
        self, target, type_name, var_name
    ):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        return (
            f"/* unsupported {target} GLSL buffer block local variable {declaration} "
            "omitted: no target-side fallback declaration emitted */"
        )

    def is_unsupported_glsl_buffer_block_name(self, name):
        return name in self.unsupported_glsl_buffer_block_variables or (
            name in self.current_unsupported_glsl_buffer_block_parameters
            or name in self.current_unsupported_glsl_buffer_block_local_variables
        )

    def unsupported_glsl_buffer_block_name_type(self, name):
        if not self.is_unsupported_glsl_buffer_block_name(name):
            return None
        return self.unsupported_glsl_buffer_block_variable_types.get(
            name
        ) or self.local_variable_types.get(name)

    def unsupported_glsl_buffer_block_expression_type(self, expr):
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_type = self.unsupported_glsl_buffer_block_expression_type(
                object_expr
            )
            member_name = str(getattr(expr, "member", ""))
            if object_type and member_name:
                return self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member_name)
            return None

        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            array_type = self.unsupported_glsl_buffer_block_expression_type(array_expr)
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type

        expr_name = self.expression_name(expr)
        if expr_name:
            block_type = self.unsupported_glsl_buffer_block_name_type(expr_name)
            if block_type:
                return block_type
        return None

    def unsupported_glsl_buffer_block_access_name(self, expr):
        expr_name = self.expression_name(expr)
        if expr_name and self.is_unsupported_glsl_buffer_block_name(expr_name):
            return expr_name
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.unsupported_glsl_buffer_block_access_name(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.unsupported_glsl_buffer_block_access_name(array_expr)
        return None

    def diagnostic_zero_value_for_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type == "bool":
            return "false"
        if mapped_type == "uint":
            return "0u"
        if mapped_type in {"float", "half", "double", "int"}:
            return "0"
        if (
            mapped_type
            and mapped_type[0].isalpha()
            and any(char.isdigit() for char in mapped_type)
        ):
            return f"{mapped_type}(0)"
        return "0"

    def unsupported_glsl_buffer_block_access_value(self, expr):
        name = self.unsupported_glsl_buffer_block_access_name(expr)
        if not name:
            return None
        value_type = (
            self.expression_result_type(expr) or self.current_expression_expected_type
        )
        fallback = self.diagnostic_zero_value_for_type(value_type)
        return (
            f"{fallback} /* unsupported Metal GLSL buffer block access {name}: "
            "no target-side fallback declaration emitted */"
        )

    def unsupported_glsl_buffer_block_assignment_diagnostic(self, target):
        name = self.unsupported_glsl_buffer_block_access_name(target)
        if not name:
            return None
        return (
            "/* unsupported Metal GLSL buffer block assignment "
            f"{name}: no target-side fallback declaration emitted */"
        )

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def global_resource_shape(self, node):
        resource_count = 1
        if hasattr(node, "var_type"):
            if (
                hasattr(node.var_type, "name")
                or hasattr(node.var_type, "element_type")
                or isinstance(node.var_type, (PointerType, ReferenceType))
            ):
                if (
                    hasattr(node.var_type, "element_type")
                    and str(type(node.var_type)).find("ArrayType") != -1
                ):
                    vtype = self.convert_type_node_to_string(node.var_type.element_type)
                    array_size = (
                        self.expression_to_string(node.var_type.size)
                        if node.var_type.size
                        else self.resource_array_size_hints.get(node.name, "")
                    )
                    resource_count = self.resource_array_count(
                        node.var_type.size if node.var_type.size else array_size
                    )
                else:
                    vtype = self.convert_type_node_to_string(node.var_type)
            else:
                vtype = str(node.var_type)
        elif hasattr(node, "vtype"):
            vtype = node.vtype
        else:
            vtype = "float"
        return vtype, resource_count

    def resource_node_name(self, node, default=None):
        if node is None:
            return default
        return getattr(
            node,
            "name",
            getattr(node, "variable_name", getattr(node, "type_name", default)),
        )

    def global_resource_binding_metadata(self, node):
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        if not var_name:
            return None

        vtype, resource_count = self.global_resource_shape(node)
        lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
        if lowered_block is not None:
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_glsl_buffer_block_variable(node, vtype):
            return None
        elif self.metal_buffer_resource_address_space(node, vtype) is not None:
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
            resource_count = self.metal_buffer_resource_binding_count(node)
        elif self.is_structured_buffer_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_acceleration_structure_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_visible_function_table_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_intersection_function_table_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_texture_or_image_resource_type(vtype):
            namespace = "texture"
            attribute_names = {"binding", "texture"}
            prefixes = ("t", "u")
        elif self.is_sampler_type(vtype):
            namespace = "sampler"
            attribute_names = {"binding", "sampler"}
            prefixes = ("s",)
        else:
            return None

        binding = self.explicit_resource_binding_index(node, attribute_names, prefixes)
        if binding is None:
            return None
        return namespace, binding, resource_count, var_name

    def reserve_explicit_global_resource_bindings(
        self, global_vars, used_bindings, source_bindings
    ):
        metadata_items = []
        for node in global_vars:
            metadata = self.global_resource_binding_metadata(node)
            if metadata is None:
                continue
            metadata_items.append((node, metadata))

        for descriptor_set in (None, 0):
            for node, metadata in metadata_items:
                if self.explicit_resource_set_index(node) != descriptor_set:
                    continue
                namespace, binding, resource_count, var_name = metadata
                self.pre_reserve_explicit_resource_binding(
                    used_bindings,
                    source_bindings,
                    namespace,
                    binding,
                    resource_count,
                    var_name,
                    node,
                )

    def next_available_resource_binding(
        self, used_bindings, namespace, binding_index, count
    ):
        count = max(count or 1, 1)
        ranges = used_bindings.get(namespace, [])
        while True:
            end = binding_index + count - 1
            conflict_end = None
            for used_start, used_end, _ in ranges:
                if binding_index <= used_end and used_start <= end:
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
            if conflict_end is None:
                return binding_index
            binding_index = conflict_end + 1

    def collect_stage_local_resource_stage_scopes(
        self, ast, target_stage, stage_local_resource_variables
    ):
        stage_scopes = {}
        stage_local_ids = {id(node) for node in stage_local_resource_variables}
        if not stage_local_ids:
            return stage_scopes

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            for variable in getattr(stage, "local_variables", []) or []:
                if id(variable) in stage_local_ids:
                    stage_scopes[id(variable)] = stage_name
        return stage_scopes

    def metal_source_binding_stage_scope(self, node):
        return self.metal_source_binding_stage_by_id.get(id(node))

    def pre_reserve_cbuffer_bindings(self, used_bindings, source_bindings):
        for cbuffer in self.cbuffer_variables:
            if self.explicit_resource_set_index(cbuffer) is not None:
                continue
            binding = self.explicit_resource_binding_index(
                cbuffer, {"binding", "buffer"}, ("b",)
            )
            if binding is None:
                continue
            self.pre_reserve_explicit_resource_binding(
                used_bindings,
                source_bindings,
                "buffer",
                binding,
                1,
                getattr(cbuffer, "name", "<anonymous>"),
                cbuffer,
            )

    def reserve_cbuffer_bindings(self, used_bindings, source_bindings):
        buffer_index = 0
        for cbuffer in self.cbuffer_variables:
            binding = self.explicit_resource_binding_index(
                cbuffer, {"binding", "buffer"}, ("b",)
            )
            if binding is None:
                binding = self.next_available_resource_binding(
                    used_bindings,
                    "buffer",
                    buffer_index,
                    1,
                )
            binding = self.reserve_or_remap_resource_binding(
                used_bindings,
                source_bindings,
                "buffer",
                binding,
                1,
                getattr(cbuffer, "name", "<anonymous>"),
                cbuffer,
                buffer_index,
            )
            self.cbuffer_binding_indices[id(cbuffer)] = binding
            buffer_index = max(buffer_index, binding + 1)
        return buffer_index

    def pre_reserve_explicit_resource_binding(
        self, used_bindings, source_bindings, namespace, binding, count, name, node
    ):
        if binding is None:
            return None
        descriptor_set = self.explicit_resource_set_index(node)
        if descriptor_set not in (None, 0):
            return None
        if descriptor_set == 0:
            self.reserve_source_resource_binding_range(
                source_bindings,
                namespace,
                descriptor_set,
                binding,
                1,
                name,
                stage_scope=self.metal_source_binding_stage_scope(node),
            )
            target_binding = self.next_available_resource_binding(
                used_bindings,
                namespace,
                binding,
                count,
            )
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                namespace,
                target_binding,
                count,
                name,
            )
            self.metal_resource_binding_indices_by_id[id(node)] = target_binding
            return target_binding
        self.reserve_resource_binding_range(
            used_bindings,
            "Metal",
            namespace,
            binding,
            count,
            name,
        )
        self.metal_resource_binding_indices_by_id[id(node)] = binding
        return binding

    def reserve_or_remap_resource_binding(
        self,
        used_bindings,
        source_bindings,
        namespace,
        binding,
        count,
        name,
        node,
        cursor,
    ):
        cached = self.metal_resource_binding_indices_by_id.get(id(node))
        if cached is not None:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                namespace,
                cached,
                count,
                name,
            )
            return cached

        requested_binding = binding
        descriptor_set = self.explicit_resource_set_index(node)
        if binding is None:
            binding = self.next_available_resource_binding(
                used_bindings,
                namespace,
                cursor,
                count,
            )

        if descriptor_set is not None:
            if requested_binding is not None:
                self.reserve_source_resource_binding_range(
                    source_bindings,
                    namespace,
                    descriptor_set,
                    requested_binding,
                    1,
                    name,
                    stage_scope=self.metal_source_binding_stage_scope(node),
                )
            target_cursor = 0 if descriptor_set != 0 else binding
            target_binding = self.next_available_resource_binding(
                used_bindings,
                namespace,
                target_cursor,
                count,
            )
        else:
            target_binding = binding

        self.reserve_resource_binding_range(
            used_bindings,
            "Metal",
            namespace,
            target_binding,
            count,
            name,
        )
        self.metal_resource_binding_indices_by_id[id(node)] = target_binding
        return target_binding

    def reserve_source_resource_binding_range(
        self,
        source_bindings,
        namespace,
        descriptor_set,
        start,
        count,
        name,
        stage_scope=None,
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        key = (namespace, descriptor_set, stage_scope)
        ranges = source_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting Metal source resource binding for '{name}': "
                    f"{self.source_resource_binding_range_label(namespace, descriptor_set, start, end, stage_scope)} "
                    f"overlaps '{used_name}' "
                    f"{self.source_resource_binding_range_label(namespace, descriptor_set, used_start, used_end, stage_scope)}"
                )
        ranges.append((start, end, name))

    def source_resource_binding_range_label(
        self, namespace, descriptor_set, start, end, stage_scope=None
    ):
        stage_label = f" {stage_scope}" if stage_scope else ""
        if start == end:
            return f"set {descriptor_set}{stage_label} {namespace}({start})"
        return f"set {descriptor_set}{stage_label} {namespace}({start}-{end})"

    def reserve_resource_binding_range(
        self, used_bindings, target, namespace, start, count, name
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        ranges = used_bindings.setdefault(namespace, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting {target} resource binding for '{name}': "
                    f"{self.resource_binding_range_label(namespace, start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.resource_binding_range_label(namespace, used_start, used_end)}"
                )
        ranges.append((start, end, name))

    def resource_binding_range_label(self, namespace, start, end):
        if start == end:
            return f"{namespace}({start})"
        return f"{namespace}({start}-{end})"

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.expression_name(array_expr)
        return None

    def texture_sampler_expression(self, texture_name, texture_arg=None):
        sampler_name = f"{texture_name}Sampler"
        if sampler_name in self.sampler_variable_names():
            index = self.array_access_index_expression(texture_arg)
            if index is not None and self.sampler_array_size(sampler_name) is not None:
                return f"{sampler_name}[{index}]"
            return sampler_name

        source_arg = self.texture_alias_sampler_source(texture_name)
        if source_arg is not None:
            member_sampler = self.struct_member_paired_sampler_expression(
                source_arg, texture_arg
            )
            if member_sampler is not None:
                return member_sampler
            source_name = self.expression_name(source_arg)
            source_sampler_name = f"{source_name}Sampler" if source_name else None
            if source_sampler_name in self.sampler_variable_names():
                index = self.array_access_index_expression(
                    source_arg
                ) or self.array_access_index_expression(texture_arg)
                if (
                    index is not None
                    and self.sampler_array_size(source_sampler_name) is not None
                ):
                    return f"{source_sampler_name}[{index}]"
                return source_sampler_name
        return self.default_sampler_expression()

    def texture_alias_sampler_source(self, texture_name):
        source_arg = self.current_texture_alias_sources.get(texture_name)
        seen = {texture_name}
        while source_arg is not None:
            source_name = self.expression_name(source_arg)
            if not source_name or source_name in seen:
                return source_arg
            next_source_arg = self.current_texture_alias_sources.get(source_name)
            if next_source_arg is None:
                return source_arg
            seen.add(source_name)
            source_arg = next_source_arg
        return None

    def struct_member_paired_sampler_expression(self, source_arg, texture_arg=None):
        member_expr = self.member_access_source_expression(source_arg)
        if member_expr is None:
            return None

        sampler_member = f"{member_expr.member}Sampler"
        sampler_node = self.struct_member_named_node(member_expr.object, sampler_member)
        if sampler_node is None or not self.struct_member_is_sampler_resource(
            sampler_node
        ):
            return None

        object_expr = self.generate_expression(member_expr.object)
        sampler_expr = f"{object_expr}.{sampler_member}"
        index = self.array_access_index_expression(
            source_arg
        ) or self.array_access_index_expression(texture_arg)
        if (
            index is not None
            and self.struct_member_array_size(sampler_node) is not None
        ):
            return f"{sampler_expr}[{index}]"
        return sampler_expr

    def array_access_index_expression(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
        if index_expr is None:
            return None
        return self.generate_expression(index_expr)

    def sampler_array_size(self, sampler_name):
        if sampler_name in self.current_sampler_parameter_array_sizes:
            return self.current_sampler_parameter_array_sizes[sampler_name]
        for sampler_variable, _, array_size in self.sampler_variables:
            if getattr(sampler_variable, "name", None) == sampler_name:
                return array_size
        return None

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        return self.texture_call_uses_explicit_sampler(args)

    def texture_call_uses_explicit_sampler(self, args):
        if len(args) < 2:
            return False
        sampler_name = self.expression_name(args[1]) or self.generate_expression(
            args[1]
        )
        if sampler_name in self.sampler_variable_names():
            return True
        arg_type = self.expression_result_type(args[1])
        return arg_type is not None and self.is_sampler_type(arg_type)

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        texture_base_name = self.expression_name(args[0]) or texture_name
        sampler_arg = (
            self.generate_expression(args[1])
            if explicit_sampler
            else self.texture_sampler_expression(texture_base_name, args[0])
        )
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, sampler_arg, coord, extra_args

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return self.struct_member_texture_resource_type(texture_arg)
        texture_type = self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )
        return texture_type or self.struct_member_texture_resource_type(texture_arg)

    def texture_sample_result_type(self, texture_arg):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        if not texture_type:
            return None
        if texture_type.startswith("depth"):
            return "float"
        component_type = "float"
        if "<" in texture_type and ">" in texture_type:
            component_type = texture_type.split("<", 1)[1].split(">", 1)[0]
            component_type = component_type.split(",", 1)[0].strip() or "float"
        return f"{component_type}4"

    def struct_member_texture_resource_type(self, texture_arg):
        member = self.struct_resource_member_node(texture_arg)
        if member is None:
            return None
        raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if raw_type is None and isinstance(member, ArrayNode):
            raw_type = getattr(member, "element_type", None)
        if self.is_array_type_node(raw_type):
            raw_type = raw_type.element_type
        elif isinstance(member, ArrayNode):
            raw_type = getattr(member, "element_type", getattr(member, "vtype", None))
        if raw_type is None:
            return None
        raw_type_name = self.type_name_string(raw_type)
        if not self.is_texture_or_image_resource_type(raw_type_name):
            return None
        return self.map_resource_type_with_format(raw_type, member)

    def struct_member_resource_array_size(self, texture_arg):
        member = self.struct_resource_member_node(texture_arg)
        if member is None:
            return None
        return self.struct_member_array_size(member)

    def struct_resource_member_node(self, texture_arg):
        expr = self.member_access_source_expression(texture_arg)
        if not isinstance(expr, MemberAccessNode):
            return None
        return self.struct_member_named_node(expr.object, str(expr.member))

    def member_access_source_expression(self, expr):
        while isinstance(expr, ArrayAccessNode):
            expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        return expr if isinstance(expr, MemberAccessNode) else None

    def struct_member_named_node(self, object_expr, member_name):
        object_type = self.expression_result_type(object_expr)
        if object_type is None:
            return None
        object_type = self.pointer_pointee_type_name(object_type) or object_type
        struct_node = self.structs_by_name.get(self.type_name_string(object_type))
        if struct_node is None:
            return None

        for member in getattr(struct_node, "members", []) or []:
            if getattr(member, "name", None) == member_name:
                return member
        return None

    def struct_member_array_size(self, member):
        if isinstance(member, ArrayNode):
            return (
                self.safe_expression_to_string(member.size)
                if member.size is not None
                else ""
            )

        raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if self.is_array_type_node(raw_type):
            return (
                self.safe_expression_to_string(raw_type.size)
                if raw_type.size is not None
                else ""
            )

        raw_type_name = self.type_name_string(raw_type)
        if raw_type_name and "[" in raw_type_name and "]" in raw_type_name:
            _, array_size = parse_array_type(raw_type_name)
            return array_size or ""
        return None

    def struct_member_is_sampler_resource(self, member):
        if isinstance(member, ArrayNode):
            raw_type = getattr(member, "element_type", getattr(member, "vtype", None))
        else:
            raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if self.is_array_type_node(raw_type):
                raw_type = raw_type.element_type
        return self.is_sampler_type(self.type_name_string(raw_type))

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        arg_type = self.expression_result_type(texture_arg)
        if arg_type is None or not self.is_texture_or_image_resource_type(arg_type):
            return None
        return self.map_resource_type_with_format(self.resource_base_type(arg_type))

    def texture_argument_raw_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name:
            raw_type = self.current_texture_parameter_raw_types.get(
                texture_name, self.texture_variable_raw_types.get(texture_name)
            )
            if raw_type is not None:
                return raw_type

        member = self.struct_resource_member_node(texture_arg)
        if member is not None:
            raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if raw_type is None and isinstance(member, ArrayNode):
                raw_type = getattr(member, "element_type", None)
            if self.is_array_type_node(raw_type):
                raw_type = raw_type.element_type
            return self.resource_base_type(self.type_name_string(raw_type))

        return self.expression_result_type(texture_arg)

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is not None and self.is_texture_or_image_resource_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"Metal texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def validate_image_resource_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_resource(texture_type):
            return
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"Metal image operation '{func_name}' requires a storage "
            f"image resource argument: {texture_name}"
        )

    def validate_image_access_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        access = self.storage_image_access_mode(
            self.texture_argument_resource_type(args[0])
        )
        if access is None:
            return
        if self.is_multisample_storage_image_resource(
            self.texture_argument_resource_type(args[0])
        ):
            return
        texture_name = expression_debug_name(args[0])
        if func_name == "imageLoad" and access == "write":
            raise ValueError(
                f"Metal image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got access::write"
            )
        if func_name == "imageStore" and access == "read":
            raise ValueError(
                f"Metal image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got access::read"
            )
        if is_image_atomic_operation(func_name) and access != "read_write":
            raise ValueError(
                f"Metal image operation '{func_name}' requires read_write "
                f"storage image access for {texture_name}: got access::{access}"
            )

    def image_atomic_value_arguments(self, func_name, args, image_type):
        has_sample = self.is_multisample_storage_image_resource(image_type)
        return shared_image_atomic_value_arguments(func_name, args, has_sample)

    def scalar_expression_kind(self, expr):
        return numeric_scalar_expression_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_value_argument_types(
        self, func_name, args, image_type, component_kind, image_format
    ):
        mismatch = image_atomic_value_kind_mismatch(
            func_name,
            self.image_atomic_value_arguments(func_name, args, image_type),
            component_kind,
            self.scalar_expression_kind,
        )
        if mismatch is None:
            return
        value_arg, value_kind = mismatch
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_atomic_value_kind_error(
                "Metal",
                func_name,
                format_label,
                component_kind,
                expression_debug_name(value_arg),
                value_kind,
            )
        )

    def scalar_expected_kind(self):
        return numeric_scalar_type_kind(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_result_type(
        self, func_name, image_type, component_kind, image_format
    ):
        expected_kind = image_atomic_result_kind_mismatch(
            self.scalar_expected_kind(), component_kind
        )
        if expected_kind is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_atomic_result_kind_error(
                "Metal", func_name, format_label, component_kind, expected_kind
            )
        )

    def validate_image_atomic_format_argument(self, func_name, args):
        if not is_image_atomic_operation(func_name) or not args:
            return
        image_type = self.texture_argument_resource_type(args[0])
        image_format = self.image_resource_format(args[0])
        component_kind = resolve_image_atomic_component_kind(
            func_name,
            image_format,
            metal_storage_image_component_type(image_type),
            "Metal",
            self.resource_base_type(image_type),
        )
        self.validate_image_atomic_value_argument_types(
            func_name,
            args,
            image_type,
            component_kind,
            image_format,
        )
        self.validate_image_atomic_result_type(
            func_name,
            image_type,
            component_kind,
            image_format,
        )

    def validate_function_image_access_arguments(self, func_name, args):
        callee_requirements = self.function_image_access_requirements.get(func_name)
        if not callee_requirements:
            return
        if not self.function_call_matches_known_signature(func_name, args):
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.storage_image_access_mode(
                self.texture_argument_resource_type(args[index])
            )
            if self.is_multisample_storage_image_resource(
                self.texture_argument_resource_type(args[index])
            ):
                continue
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = image_access_requirement_label(
                required_access, read_write_label="read_write"
            )
            raise ValueError(
                f"Metal function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got access::{actual_access}"
            )

    def split_metal_array_resource_type(self, type_name):
        type_name = str(type_name or "").strip()
        if not type_name.startswith("array<") or not type_name.endswith(">"):
            return None

        body = type_name[len("array<") : -1]
        depth = 0
        for index, char in enumerate(body):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                return body[:index].strip(), body[index + 1 :].strip()
        return None

    def metal_array_resource_type(self, element_type, array_size):
        return f"array<{element_type}, {array_size or '1'}>"

    def function_parameter_resource_type(self, func_name, parameter):
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        param_name = getattr(parameter, "name", None)
        function_hints = self.function_resource_array_size_hints.get(func_name, {})

        if self.is_array_type_node(raw_type):
            base_type = self.type_name_string(raw_type.element_type)
            if not self.is_resource_parameter_type(base_type):
                return None
            array_size = (
                self.safe_expression_to_string(raw_type.size)
                if raw_type.size is not None
                else function_hints.get(param_name, "")
            )
            return self.metal_array_resource_type(
                self.map_resource_type_with_format(base_type, parameter),
                array_size,
            )

        type_string = self.type_name_string(raw_type)
        if "[" in type_string and "]" in type_string:
            base_type, array_size = parse_array_type(type_string)
            if not self.is_resource_parameter_type(base_type):
                return None
            return self.metal_array_resource_type(
                self.map_resource_type_with_format(base_type, parameter),
                (
                    function_hints.get(param_name, "")
                    if array_size is None
                    else array_size
                ),
            )

        if not self.is_resource_parameter_type(type_string):
            return None
        return self.map_resource_type_with_format(type_string, parameter)

    def function_argument_resource_type(self, arg):
        resource_type = self.texture_argument_resource_type(arg)
        if resource_type is not None:
            array_size = self.texture_argument_resource_array_size(arg)
            if array_size is not None:
                return self.metal_array_resource_type(resource_type, array_size)
            return resource_type

        resource_type = self.sampler_argument_resource_type(arg)
        if resource_type is None:
            return None
        array_size = self.sampler_argument_resource_array_size(arg)
        if array_size is not None:
            return self.metal_array_resource_type(resource_type, array_size)
        return resource_type

    def normalize_function_resource_compatibility_type(self, resource_type):
        array_resource = self.split_metal_array_resource_type(resource_type)
        if array_resource is not None:
            element_type, array_size = array_resource
            return self.metal_array_resource_type(
                self.normalize_function_resource_compatibility_type(element_type),
                self.normalize_function_resource_array_size(array_size),
            )
        if self.is_storage_image_resource(resource_type):
            return self.storage_image_access_agnostic_type(resource_type)
        return self.resource_base_type(resource_type)

    def normalize_function_resource_array_size(self, array_size):
        literal_size = self.literal_int_resource_array_size(array_size)
        if literal_size is not None:
            return str(literal_size)
        return str(array_size or "")

    def literal_int_resource_array_size(self, array_size):
        value = self.literal_int_value(array_size, self.literal_int_constants)
        if value is not None:
            return value

        if not isinstance(array_size, str):
            return None
        try:
            parsed = py_ast.parse(array_size, mode="eval")
        except SyntaxError:
            return None
        return self.literal_int_python_expression_value(parsed.body)

    def literal_int_python_expression_value(self, expr):
        if isinstance(expr, py_ast.Constant) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, py_ast.Name):
            return self.literal_int_constants.get(expr.id)
        if isinstance(expr, py_ast.UnaryOp):
            operand = self.literal_int_python_expression_value(expr.operand)
            if operand is None:
                return None
            if isinstance(expr.op, py_ast.UAdd):
                return operand
            if isinstance(expr.op, py_ast.USub):
                return -operand
            return None
        if isinstance(expr, py_ast.BinOp):
            left = self.literal_int_python_expression_value(expr.left)
            right = self.literal_int_python_expression_value(expr.right)
            if left is None or right is None:
                return None
            if isinstance(expr.op, py_ast.Add):
                return left + right
            if isinstance(expr.op, py_ast.Sub):
                return left - right
            if isinstance(expr.op, py_ast.Mult):
                return left * right
        return None

    def resource_type_requires_function_resource_compatibility(self, resource_type):
        array_resource = self.split_metal_array_resource_type(resource_type)
        if array_resource is not None:
            element_type, _ = array_resource
            return self.resource_type_requires_function_resource_compatibility(
                element_type
            )

        base_type = self.resource_base_type(resource_type)
        return self.is_sampler_type(base_type) or str(base_type).startswith("texture")

    def validate_function_resource_argument_types(self, func_name, args):
        parameter_nodes = self.function_parameter_nodes.get(func_name)
        if not parameter_nodes:
            return
        if not self.function_call_matches_known_signature(func_name, args):
            return

        for index, parameter in enumerate(parameter_nodes):
            if index >= len(args):
                return
            expected_type = self.function_parameter_resource_type(func_name, parameter)
            if expected_type is None:
                continue

            actual_type = self.function_argument_resource_type(args[index])
            if not self.resource_type_requires_function_resource_compatibility(
                expected_type
            ):
                continue

            if self.normalize_function_resource_compatibility_type(
                expected_type
            ) == self.normalize_function_resource_compatibility_type(actual_type):
                continue

            actual_name = expression_debug_name(args[index])
            actual_type_label = actual_type or self.type_name_string(
                self.expression_result_type(args[index])
            )
            raise ValueError(
                f"Metal function call '{func_name}' requires resource parameter "
                f"{getattr(parameter, 'name', None)} of type {expected_type}: "
                f"argument {actual_name} has {actual_type_label or 'non-resource'}"
            )

    def validate_integer_coordinate_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        coord_type = self.expression_result_type(args[1])
        if coord_type is None or self.is_integer_coordinate_type(coord_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "Metal",
                "resource",
                func_name,
                "an integer",
                "coordinate",
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_coordinate_dimension_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[1])
        coord_dimension = integer_coordinate_dimension_from_type_name(
            self.type_name_string(coord_type), self.map_type
        )
        if coord_dimension is None or coord_dimension == expected_dimension:
            return
        raise ValueError(
            operation_dimension_argument_error(
                "Metal",
                "resource",
                func_name,
                expected_dimension,
                "integer",
                "coordinate",
                self.resource_base_type(texture_type),
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_offset_dimension_argument(self, func_name, args):
        offset_indices = texture_offset_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not offset_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_offset_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for offset_index in offset_indices:
            offset_type = self.expression_result_type(args[offset_index])
            if offset_type is None:
                continue
            if not self.is_integer_coordinate_type(offset_type):
                raise ValueError(
                    operation_argument_type_error(
                        "Metal",
                        "resource",
                        func_name,
                        "an integer",
                        "offset",
                        expression_debug_name(args[offset_index]),
                        self.type_name_string(offset_type),
                    )
                )
            offset_dimension = integer_coordinate_dimension_from_type_name(
                self.type_name_string(offset_type), self.map_type
            )
            if offset_dimension is None or offset_dimension == expected_dimension:
                continue
            raise ValueError(
                operation_dimension_argument_error(
                    "Metal",
                    "resource",
                    func_name,
                    expected_dimension,
                    "integer",
                    "offset",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[offset_index]),
                    self.type_name_string(offset_type),
                )
            )

    def gradient_argument_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def query_lod_coordinate_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def validate_query_lod_coordinate_argument(self, func_name, args):
        coord_index = texture_query_lod_coordinate_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if coord_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_query_lod_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[coord_index])
        if coord_type is None:
            return
        coord_dimension = self.query_lod_coordinate_dimension(coord_type)
        if coord_dimension is None:
            raise ValueError(
                texture_query_lod_coordinate_type_error(
                    "Metal",
                    func_name,
                    expression_debug_name(args[coord_index]),
                    self.type_name_string(coord_type),
                )
            )
        if coord_dimension == expected_dimension:
            return
        raise ValueError(
            texture_query_lod_coordinate_dimension_error(
                "Metal",
                func_name,
                expected_dimension,
                self.resource_base_type(texture_type),
                expression_debug_name(args[coord_index]),
                self.type_name_string(coord_type),
            )
        )

    def validate_gradient_dimension_arguments(self, func_name, args):
        gradient_indices = texture_gradient_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not gradient_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_gradient_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for gradient_index in gradient_indices:
            gradient_type = self.expression_result_type(args[gradient_index])
            if gradient_type is None:
                continue
            gradient_dimension = self.gradient_argument_dimension(gradient_type)
            if gradient_dimension is None:
                raise ValueError(
                    operation_argument_type_error(
                        "Metal",
                        "resource",
                        func_name,
                        "a floating",
                        "gradient",
                        expression_debug_name(args[gradient_index]),
                        self.type_name_string(gradient_type),
                    )
                )
            if gradient_dimension == expected_dimension:
                continue
            raise ValueError(
                operation_dimension_argument_error(
                    "Metal",
                    "resource",
                    func_name,
                    expected_dimension,
                    "floating",
                    "gradient",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[gradient_index]),
                    self.type_name_string(gradient_type),
                )
            )

    def is_scalar_floating_type(self, vtype):
        return is_floating_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_numeric_type(self, vtype):
        return is_numeric_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_integer_type(self, vtype):
        return is_integer_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def texture_argument_diagnostic_type(self, arg):
        sampler_names = {
            sampler_variable.name for sampler_variable, _, _ in self.sampler_variables
        } | self.current_sampler_parameters
        return shared_texture_argument_diagnostic_type(
            arg,
            self.texture_resource_type,
            self.expression_name,
            self.expression_result_type,
            sampler_names,
        )

    def validate_compare_argument(self, func_name, args):
        compare_index = texture_compare_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if compare_index is None:
            return
        compare_type = self.expression_result_type(args[compare_index])
        if compare_type is None or self.is_scalar_floating_type(compare_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "Metal",
                "texture compare",
                func_name,
                "a scalar floating",
                "compare",
                expression_debug_name(args[compare_index]),
                self.type_name_string(compare_type),
            )
        )

    def validate_lod_argument(self, func_name, args):
        lod_index = texture_lod_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if lod_index is None:
            return
        lod_type = self.texture_argument_diagnostic_type(args[lod_index])
        if lod_type is None or self.is_scalar_numeric_type(lod_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "Metal",
                "texture LOD",
                func_name,
                "a scalar numeric",
                "lod",
                expression_debug_name(args[lod_index]),
                self.type_name_string(lod_type),
            )
        )

    def validate_bias_argument(self, func_name, args):
        bias_index = texture_bias_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if bias_index is None:
            return
        bias_type = self.texture_argument_diagnostic_type(args[bias_index])
        if bias_type is None or self.is_scalar_numeric_type(bias_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "Metal",
                "texture bias",
                func_name,
                "a scalar numeric",
                "bias",
                expression_debug_name(args[bias_index]),
                self.type_name_string(bias_type),
            )
        )

    def validate_mip_level_argument(self, func_name, args):
        level_index = texture_mip_level_argument_index(func_name, len(args))
        if level_index is None:
            return
        level_type = self.texture_argument_diagnostic_type(args[level_index])
        if level_type is None or self.is_scalar_integer_type(level_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "Metal",
                "resource",
                func_name,
                "a scalar integer",
                "mip/sample level",
                expression_debug_name(args[level_index]),
                self.type_name_string(level_type),
            )
        )

    def validate_sample_index_argument(self, func_name, args):
        sample_index = texture_sample_index_argument_index(func_name, len(args))
        if sample_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if not self.is_multisample_texture_resource(texture_type):
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        if sample_type is None or self.is_scalar_integer_type(sample_type):
            return
        raise ValueError(
            texture_multisample_sample_type_error(
                "Metal",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_image_multisample_arguments(self, func_name, args):
        if not args:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        sample_index = image_multisample_sample_argument_index(
            func_name,
            len(args),
            self.is_multisample_storage_image_resource(texture_type),
            "Metal",
        )
        if sample_index is None:
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        sample_type = image_multisample_sample_type_mismatch(
            sample_type, self.is_scalar_integer_type
        )
        if sample_type is None:
            return
        raise ValueError(
            image_multisample_sample_type_error(
                "Metal",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_gather_component_argument(self, func_name, args):
        component_index = texture_gather_component_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if component_index is None:
            return
        component_type = self.texture_argument_diagnostic_type(args[component_index])
        if component_type is None or self.is_scalar_integer_type(component_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "Metal",
                "texture gather",
                func_name,
                "a scalar integer",
                "component",
                expression_debug_name(args[component_index]),
                self.type_name_string(component_type),
            )
        )

    def validate_texture_call_arity(self, func_name, args):
        if self.is_packed_shadow_texture_sample_call(func_name, args):
            return
        validate_texture_operation_arity(
            "Metal",
            func_name,
            args,
            self.texture_resource_operation_names(),
            self.texture_call_uses_explicit_sampler,
        )

    def texture_resource_operation_names(self):
        return texture_image_resource_operation_names(IMAGE_RESOURCE_INTRINSIC_NAMES)

    def image_resource_format(self, texture_arg):
        image_format = image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )
        if image_format is not None:
            return image_format
        member = self.struct_resource_member_node(texture_arg)
        if member is None:
            return None
        return explicit_image_format(member, self.attribute_value_to_string)

    def is_array_texture_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return False
        return texture_type.startswith(
            (
                "texture1d_array<",
                "texture2d_array<",
                "depth2d_array<",
                "texturecube_array<",
                "depthcube_array<",
            )
        )

    def is_multisample_texture_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return False
        return texture_type.startswith(("texture2d_ms<", "texture2d_ms_array<"))

    def is_multisample_storage_image_resource(self, texture_type):
        texture_type = self.storage_image_access_agnostic_type(texture_type)
        return texture_type.startswith(("texture2d_ms<", "texture2d_ms_array<"))

    def is_storage_image_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return is_metal_storage_image_resource(texture_type)

    def storage_image_access_mode(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if "access::read_write>" in texture_type:
            return "read_write"
        if "access::read>" in texture_type:
            return "read"
        if "access::write>" in texture_type:
            return "write"
        return None

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def array_texture_coordinate_parts(self, coord):
        coord_xy = self.vector_component(coord, "xy")
        layer = f"uint({self.vector_component(coord, 'z')})"
        return coord_xy, layer

    def cube_array_texture_coordinate_parts(self, coord):
        coord_xyz = self.vector_component(coord, "xyz")
        layer = f"uint({self.vector_component(coord, 'w')})"
        return coord_xyz, layer

    def texture_coordinate_parts(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("texture1d_array<"):
            coord_x = self.vector_component(coord, "x")
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer
        if texture_type.startswith(("texturecube_array<", "depthcube_array<")):
            return self.cube_array_texture_coordinate_parts(coord)
        return self.array_texture_coordinate_parts(coord)

    def texture_read_coordinate_components(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("texture1d_array<"):
            coord_x = self.vector_component(coord, "x")
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer, 1
        if texture_type.startswith(
            ("texture2d_array<", "depth2d_array<", "texture2d_ms_array<")
        ):
            coord_xy = self.vector_component(coord, "xy")
            layer = f"uint({self.vector_component(coord, 'z')})"
            return coord_xy, layer, 2
        if texture_type.startswith("texture1d<"):
            return coord, None, 1
        if texture_type.startswith("texture3d<"):
            return coord, None, 3
        return coord, None, 2

    def texture_read_coordinate_parts(self, texture_type, coord):
        texel_coord, layer, dimensions = self.texture_read_coordinate_components(
            texture_type, coord
        )
        return self.unsigned_coordinate_expression(texel_coord, dimensions), layer

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("Metal", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

    def texture_gradient_options(self, texture_type, ddx, ddy):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith(
            ("texturecube<", "depthcube<", "texturecube_array<", "depthcube_array<")
        ):
            return f"gradientcube({ddx}, {ddy})"
        if texture_type.startswith("texture3d<"):
            return f"gradient3d({ddx}, {ddy})"
        return f"gradient2d({ddx}, {ddy})"

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        is_sampled_texture = not self.is_storage_image_resource(texture_type)
        is_texture2d = is_sampled_texture and texture_type.startswith("texture2d<")
        is_texture2d_array = is_sampled_texture and texture_type.startswith(
            "texture2d_array<"
        )
        is_texture3d = is_sampled_texture and texture_type.startswith("texture3d<")
        is_texturecube = is_sampled_texture and texture_type.startswith("texturecube<")
        is_texturecube_array = is_sampled_texture and texture_type.startswith(
            "texturecube_array<"
        )
        is_depth2d = texture_type.startswith("depth2d<")
        is_depth2d_array = texture_type.startswith("depth2d_array<")
        return {
            "texture_type": texture_type,
            "gather": (
                is_texture2d
                or is_texture2d_array
                or is_texturecube
                or is_texturecube_array
            ),
            "gather_offset": is_texture2d or is_texture2d_array,
            "sample_offset": is_texture2d or is_texture2d_array or is_texture3d,
            "projected_offset": is_texture2d or is_texture2d_array or is_texture3d,
            "compare_offset": is_depth2d or is_depth2d_array,
            "gather_compare_offset": is_depth2d or is_depth2d_array,
        }

    def texture_gather_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_sample_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

    def is_texture1d_sample_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return False
        return texture_type.startswith(("texture1d<", "texture1d_array<"))

    def unsupported_texture1d_sampling_option_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return (
            "/* unsupported Metal texture sampling: "
            f"{func_name} on {texture_type} supports only implicit sampling */ "
            "float4(0.0)"
        )

    def unsupported_texture1d_texel_fetch_lod_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return (
            "/* unsupported Metal texel fetch: "
            f"{func_name} on {texture_type} requires a compile-time literal "
            "mip level */ float4(0.0)"
        )

    def unsupported_texture1d_size_lod_call(self, texture_type, return_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = "0" if return_type == "int" else f"{return_type}(0)"
        return (
            "/* unsupported Metal texture size query: textureSize on "
            f"{texture_type} requires a compile-time literal mip level */ "
            f"{zero_value}"
        )

    def metal_texture1d_read_lod_argument(self, lod_arg):
        if lod_arg is None:
            return "uint(0)"
        value = self.literal_int_value(lod_arg, self.literal_int_constants)
        if value is None:
            return None
        return f"uint({value})"

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        return unsupported_texture_offset_call_expression("Metal", func_name, reason)

    def texture_sample_offset_coord_args(self, texture_type, coord):
        if self.is_array_texture_resource(texture_type):
            return self.texture_coordinate_parts(texture_type, coord)
        return (coord,)

    def generate_texture_sample_offset_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not self.texture_sample_supports_offset(texture_type):
            return self.unsupported_texture_sample_offset_call(
                func_name, texture_sample_offset_capability_error("Metal")
            )

        coord_args = self.texture_sample_offset_coord_args(texture_type, coord)

        if is_texture_sample_basic_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            offset = self.generate_expression(extra_args[0])
            args = [sampler_arg] + list(coord_args)
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                args.append(f"bias({bias})")
            args.append(offset)
            return f"{texture_name}.sample({', '.join(args)})"

        if is_texture_sample_lod_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [f"level({lod})", offset]
            return f"{texture_name}.sample({', '.join(args)})"

        if is_texture_sample_grad_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            args = [sampler_arg] + list(coord_args) + [gradient_options, offset]
            return f"{texture_name}.sample({', '.join(args)})"

        return self.unsupported_texture_sample_offset_call(
            func_name, unsupported_texture_offset_operation_error()
        )

    def unsupported_texture_projected_call(self, func_name, reason):
        return unsupported_projected_texture_call_expression("Metal", func_name, reason)

    def projected_texture_coord(self, texture_arg, coord_arg, coord):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        specs = {
            "texture1d<float>": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "texture2d<float>": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "texture2d_array<float>": {
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "texture3d<float>": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
            "texturecube<float>": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
        }
        texture_specs = specs.get(texture_type)
        if texture_specs is None:
            return None
        coord_spec = texture_specs.get(coord_type)
        if coord_spec is None:
            return None
        numerator, divisor = coord_spec
        projected_coord = (
            f"{self.vector_component(coord, numerator)} / "
            f"{self.vector_component(coord, divisor)}"
        )
        if texture_type == "texture2d_array<float>":
            return f"{projected_coord}, uint({self.vector_component(coord, 'z')})"
        return projected_coord

    def projected_texture_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["projected_offset"]

    def generate_texture_projected_call(
        self,
        func_name,
        texture_name,
        sampler_arg,
        coord,
        extra_args,
        texture_type,
        args,
    ):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        projected_coord = self.projected_texture_coord(
            args[0], args[coord_index], coord
        )
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires 1D, 2D, or 3D projection coordinates"
            )

        if is_projected_texture_basic_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if extra_args and self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "bias is not supported for Metal 1D textures"
                )
            if not extra_args:
                return f"{texture_name}.sample({sampler_arg}, {projected_coord})"
            if len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.sample("
                    f"{sampler_arg}, {projected_coord}, bias({bias}))"
                )

        if is_projected_texture_basic_offset_operation(func_name):
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, projected_texture_offset_capability_error()
                )
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if len(extra_args) == 1:
                offset = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.sample({sampler_arg}, {projected_coord}, {offset})"
                )
            if len(extra_args) == 2:
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.sample("
                    f"{sampler_arg}, {projected_coord}, bias({bias}), {offset})"
                )

        if is_projected_texture_lod_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "explicit LOD is not supported for Metal 1D textures"
                )
            lod = self.generate_expression(extra_args[0])
            return (
                f"{texture_name}.sample({sampler_arg}, {projected_coord}, level({lod}))"
            )

        if is_projected_texture_lod_offset_operation(func_name):
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, projected_texture_offset_capability_error()
                )
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, level({lod}), {offset})"
            )

        if is_projected_texture_grad_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "gradients are not supported for Metal 1D textures"
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, {gradient_options})"
            )

        if is_projected_texture_grad_offset_operation(func_name):
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, projected_texture_offset_capability_error()
                )
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, {gradient_options}, {offset})"
            )

        return self.unsupported_texture_projected_call(
            func_name, unsupported_projected_texture_operation_error()
        )

    def is_array_expression(self, node):
        type_name = self.expression_result_type(node)
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offsets_name = self.generate_expression(extra_args[0])
            offset_args = [f"{offsets_name}[{index}]" for index in range(4)]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_component_option(self, component_arg):
        if component_arg is None:
            return None

        components = {
            0: "component::x",
            1: "component::y",
            2: "component::z",
            3: "component::w",
        }
        return components.get(self.literal_int_value(component_arg))

    def texture_gather_coord_args(self, texture_type, coord):
        if self.is_array_texture_resource(texture_type):
            coord_part, layer = self.texture_coordinate_parts(texture_type, coord)
            return [coord_part, layer]
        return [coord]

    def texture_gather_call_expression(
        self,
        texture_name,
        sampler_arg,
        coord_args,
        offset_arg=None,
        component=None,
        default_offset_for_component=False,
    ):
        args = [sampler_arg] + coord_args
        if offset_arg is not None:
            args.append(offset_arg)
        elif component is not None and default_offset_for_component:
            args.append("int2(0)")
        if component is not None:
            args.append(component)
        return f"{texture_name}.gather({', '.join(args)})"

    def texture_gather_offsets_expression(
        self, texture_name, sampler_arg, coord_args, offset_args, component
    ):
        component_suffixes = ("x", "y", "z", "w")
        component_values = []
        for index, offset_arg in enumerate(offset_args):
            gather = self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                self.generate_expression(offset_arg),
                component,
                default_offset_for_component=True,
            )
            component_values.append(f"{gather}.{component_suffixes[index]}")
        return f"float4({', '.join(component_values)})"

    def texture_gather_dynamic_component_expression(
        self, build_expression, component_expr
    ):
        component_options = (
            "component::x",
            "component::y",
            "component::z",
            "component::w",
        )
        component_calls = [
            build_expression(component) for component in component_options
        ]
        return (
            f"({component_expr} == 0 ? {component_calls[0]} : "
            f"{component_expr} == 1 ? {component_calls[1]} : "
            f"{component_expr} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return unsupported_texture_gather_call_expression("Metal", func_name, reason)

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        return unsupported_multisample_texture_call_vector_expression(
            "Metal", func_name, texture_type
        )

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        return unsupported_multisample_texture_compare_scalar_expression(
            "Metal", func_name, texture_type
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        return unsupported_multisample_texture_gather_compare_vector_expression(
            "Metal", func_name, texture_type
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        return unsupported_multisample_texture_query_lod_expression(
            "Metal", texture_type
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_levels_expression("Metal", texture_type)

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_lod_expression("Metal", texture_type)

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_resource(texture_type):
            return None

        texture_type = self.resource_base_type(texture_type)
        if is_storage_image_texture_comparison_operation(func_name):
            return unsupported_storage_image_texture_comparison_scalar_expression(
                "Metal", func_name, texture_type
            )

        if is_storage_image_texture_operation(func_name):
            return unsupported_storage_image_texture_operation_vector_expression(
                "Metal", func_name, texture_type
            )

        return None

    def is_cube_texture_resource(self, texture_type):
        return texture_type in {
            "texturecube<float>",
            "texturecube_array<float>",
            "depthcube<float>",
            "depthcube_array<float>",
        }

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        return unsupported_cube_texel_fetch_expression("Metal", func_name, texture_type)

    def generate_texture_gather_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)
        if is_texture_gather_basic_operation(
            func_name
        ) and not self.texture_gather_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_capability_error()
            )

        coord_args = self.texture_gather_coord_args(texture_type, coord)
        supports_offset = self.texture_gather_supports_offset(texture_type)
        offset_args = []
        component_arg = None

        if is_texture_gather_basic_operation(func_name):
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_component_count_error()
                )
            if extra_args:
                component_arg = extra_args[0]
        elif is_texture_gather_single_offset_operation(func_name):
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_argument_count_error()
                )
            if not supports_offset:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_capability_error()
                )
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        elif is_texture_gather_multi_offset_operation(func_name):
            if not supports_offset:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_capability_error()
                )
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offsets_argument_count_error()
                )
        else:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_operation_error()
            )

        component = self.texture_gather_component_option(component_arg)
        if component is not None or component_arg is None:
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name, sampler_arg, coord_args, offset_args, component
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            return self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                offset_arg,
                component,
                default_offset_for_component=supports_offset,
            )

        if self.literal_int_value(component_arg) is not None:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_component_literal_error()
            )

        component_expr = self.generate_expression(component_arg)
        if is_texture_gather_multi_offset_operation(func_name):
            return self.texture_gather_dynamic_component_expression(
                lambda option: self.texture_gather_offsets_expression(
                    texture_name, sampler_arg, coord_args, offset_args, option
                ),
                component_expr,
            )

        offset_arg = self.generate_expression(offset_args[0]) if offset_args else None
        return self.texture_gather_dynamic_component_expression(
            lambda option: self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                offset_arg,
                option,
                default_offset_for_component=supports_offset,
            ),
            component_expr,
        )

    def texture_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

    def unsupported_texture_compare_call(self, func_name, reason):
        return unsupported_texture_compare_scalar_expression("Metal", func_name, reason)

    def is_depth_texture_resource(self, texture_type):
        return self.resource_base_type(texture_type).startswith("depth")

    def is_packed_shadow_texture_sample_call(self, func_name, args):
        if func_name not in {
            "texture",
            "textureLod",
            "textureOffset",
            "textureLodOffset",
        }:
            return False
        if not args:
            return False
        return self.is_depth_texture_resource(
            self.texture_argument_resource_type(args[0])
        )

    def packed_shadow_compare_coord_args(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("depth2d_array<"):
            return [
                self.vector_component(coord, "xy"),
                f"uint({self.vector_component(coord, 'z')})",
            ], self.vector_component(coord, "w")
        if texture_type.startswith("depth2d<"):
            return [self.vector_component(coord, "xy")], self.vector_component(
                coord, "z"
            )
        if texture_type.startswith("depthcube_array<"):
            return [
                self.vector_component(coord, "xyz"),
                f"uint({self.vector_component(coord, 'w')})",
            ], None
        if texture_type.startswith("depthcube<"):
            return [self.vector_component(coord, "xyz")], self.vector_component(
                coord, "w"
            )
        return None, None

    def generate_packed_shadow_texture_sample_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not self.is_depth_texture_resource(texture_type):
            return None

        coord_args, packed_compare = self.packed_shadow_compare_coord_args(
            texture_type, coord
        )
        if coord_args is None:
            return self.unsupported_texture_compare_call(
                func_name, "requires supported shadow texture coordinates"
            )

        is_cube_array = self.resource_base_type(texture_type).startswith(
            "depthcube_array<"
        )
        sample_options = []
        trailing_args = []

        if func_name == "texture":
            if is_cube_array:
                if len(extra_args) not in {1, 2}:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "cube-array shadow sampling requires compare and optional bias arguments",
                    )
                compare = self.generate_expression(extra_args[0])
                if len(extra_args) == 2:
                    bias = self.generate_expression(extra_args[1])
                    sample_options.append(f"bias({bias})")
            else:
                if len(extra_args) > 1:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "packed shadow sampling accepts at most one bias argument",
                    )
                compare = packed_compare
                if extra_args:
                    bias = self.generate_expression(extra_args[0])
                    sample_options.append(f"bias({bias})")
        elif func_name == "textureLod":
            if is_cube_array:
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "cube-array shadow LOD sampling requires compare and lod arguments",
                    )
                compare = self.generate_expression(extra_args[0])
                lod_arg = extra_args[1]
            else:
                if len(extra_args) != 1:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "packed shadow LOD sampling requires one lod argument",
                    )
                compare = packed_compare
                lod_arg = extra_args[0]
            sample_options.append(f"level({self.generate_expression(lod_arg)})")
        elif func_name == "textureOffset":
            if is_cube_array or not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "packed shadow offset sampling requires offset and optional bias arguments",
                )
            compare = packed_compare
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                sample_options.append(f"bias({bias})")
            trailing_args.append(self.generate_expression(extra_args[0]))
        elif func_name == "textureLodOffset":
            if is_cube_array or not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "packed shadow LOD offset sampling requires lod and offset arguments",
                )
            compare = packed_compare
            sample_options.append(f"level({self.generate_expression(extra_args[0])})")
            trailing_args.append(self.generate_expression(extra_args[1]))
        else:
            return None

        args = [sampler_arg] + coord_args + [compare] + sample_options + trailing_args
        return f"{texture_name}.sample_compare({', '.join(args)})"

    def texture_compare_projected_coord_args(self, texture_type, coord_arg, coord):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "depth2d<float>":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            return [f"{self.vector_component(coord, 'xy')} / {divisor}"]

        if texture_type == "depthcube<float>":
            if coord_type not in {"vec4", "float4"}:
                return None
            return [
                f"{self.vector_component(coord, 'xyz')} / "
                f"{self.vector_component(coord, 'w')}"
            ]

        if texture_type != "depth2d_array<float>" or coord_type not in {
            "vec4",
            "float4",
        }:
            return None

        projected_coord = (
            f"{self.vector_component(coord, 'xy')} / "
            f"{self.vector_component(coord, 'w')}"
        )
        layer = f"uint({self.vector_component(coord, 'z')})"
        return [projected_coord, layer]

    def generate_texture_compare_call(
        self,
        func_name,
        texture_name,
        sampler_arg,
        coord,
        extra_args,
        texture_type,
        args=None,
    ):
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, texture_compare_argument_error()
            )

        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if is_projected_texture_compare_operation(func_name):
            coord_index = 2 if self.is_explicit_sampler_argument(args or []) else 1
            coord_arg = (args or [None, None])[coord_index]
            coord_args = self.texture_compare_projected_coord_args(
                texture_type, coord_arg, coord
            )
            if coord_args is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_projected_coordinate_error("Metal")
                )
            projected_args = [sampler_arg] + coord_args + [compare]

            if is_texture_compare_basic_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return f"{texture_name}.sample_compare({', '.join(projected_args)})"

            if is_texture_compare_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("Metal")
                    )
                offset = self.generate_expression(extra_args[1])
                args = projected_args + [offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_lod_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                lod = self.generate_expression(extra_args[1])
                args = projected_args + [f"level({lod})"]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_lod_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("Metal")
                    )
                lod = self.generate_expression(extra_args[1])
                offset = self.generate_expression(extra_args[2])
                args = projected_args + [f"level({lod})", offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_grad_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
                args = projected_args + [gradient_options]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_grad_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("Metal")
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
                offset = self.generate_expression(extra_args[3])
                args = projected_args + [gradient_options, offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            return self.unsupported_texture_compare_call(
                func_name, unsupported_texture_compare_operation_error(projected=True)
            )

        coord_args = (
            self.texture_coordinate_parts(texture_type, coord)
            if self.is_array_texture_resource(texture_type)
            else (coord,)
        )

        if is_texture_compare_basic_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            args = [sampler_arg] + list(coord_args) + [compare]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            offset = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [compare, offset]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_lod_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            lod = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [compare, f"level({lod})"]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_lod_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            lod = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            args = (
                [sampler_arg]
                + list(coord_args)
                + [
                    compare,
                    f"level({lod})",
                    offset,
                ]
            )
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_grad_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            args = [sampler_arg] + list(coord_args) + [compare, gradient_options]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_grad_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            offset = self.generate_expression(extra_args[3])
            args = (
                [sampler_arg]
                + list(coord_args)
                + [
                    compare,
                    gradient_options,
                    offset,
                ]
            )
            return f"{texture_name}.sample_compare({', '.join(args)})"

        return self.unsupported_texture_compare_call(
            func_name, unsupported_texture_compare_operation_error()
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return unsupported_texture_gather_compare_call_expression(
            "Metal", func_name, reason
        )

    def generate_texture_gather_compare_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_argument_error()
            )

        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        coord_args = self.texture_gather_coord_args(texture_type, coord)
        if func_name == "textureGatherCompare":
            count_error = texture_gather_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_gather_compare_call(
                    func_name, count_error
                )
            args = [sampler_arg] + coord_args + [compare]
            return f"{texture_name}.gather_compare({', '.join(args)})"

        count_error = texture_gather_compare_extra_argument_count_error(
            func_name, len(extra_args)
        )
        if count_error:
            return self.unsupported_texture_gather_compare_call(func_name, count_error)
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("Metal")
            )
        offset = self.generate_expression(extra_args[1])
        args = [sampler_arg] + coord_args + [compare, offset]
        return f"{texture_name}.gather_compare({', '.join(args)})"

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_argument_resource_type(texture_arg)
        raw_type = self.texture_argument_raw_resource_type(texture_arg)
        return {
            "texture_type": texture_type,
            "storage_image": self.is_storage_image_resource(texture_type),
            "multisample": (
                self.is_multisample_texture_resource(texture_type)
                or self.is_multisample_storage_image_resource(texture_type)
            ),
            "size_descriptor": self.texture_query_size_descriptor_for_argument(
                texture_type, raw_type
            ),
        }

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        query_descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = query_descriptor["texture_type"]
        size_descriptor = query_descriptor["size_descriptor"]
        if size_descriptor is None:
            return None

        if self.is_texture1d_sample_resource(texture_type):
            lod_arg_string = self.metal_texture1d_read_lod_argument(lod_arg)
            if lod_arg_string is None:
                return self.unsupported_texture1d_size_lod_call(
                    texture_type, size_descriptor["return_type"]
                )
        else:
            lod = self.generate_expression(lod_arg) if lod_arg is not None else "0"
            lod_arg_string = f"uint({lod})"
        return self.texture_query_size_descriptor_expression(
            texture_name, size_descriptor, lod_arg_string
        )

    def texture_query_size_descriptor_expression(
        self, texture_name, descriptor, lod_arg_string
    ):
        dimension_expressions = [
            self.texture_query_size_dimension_expression(
                texture_name, method, use_lod, lod_arg_string
            )
            for method, use_lod in descriptor["dimensions"]
        ]
        if descriptor["return_type"] == "int":
            return f"int({dimension_expressions[0]})"
        return f"{descriptor['return_type']}({', '.join(dimension_expressions)})"

    def texture_query_size_dimension_expression(
        self, texture_name, method, use_lod, lod_arg_string
    ):
        args = lod_arg_string if use_lod else ""
        return f"{texture_name}.{method}({args})"

    def texture_query_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return self.storage_image_size_descriptor(texture_type)
        return self.sampled_texture_size_descriptor(texture_type)

    def texture_query_size_descriptor_for_argument(self, texture_type, raw_type=None):
        raw_base_type = self.resource_base_type(raw_type)
        if raw_base_type == "imageCube":
            return resource_query_method_size_descriptor(
                "int2", (("get_width", False), ("get_height", False))
            )
        return self.texture_query_size_descriptor(texture_type)

    def storage_image_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        make_descriptor = resource_query_method_size_descriptor
        if texture_type.startswith("texture1d_array<"):
            return make_descriptor(
                "int2", (("get_width", False), ("get_array_size", False))
            )
        if texture_type.startswith("texture1d<"):
            return make_descriptor("int", (("get_width", False),))
        if texture_type.startswith("texture2d_array<"):
            return make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            )
        if texture_type.startswith("texture2d_ms_array<"):
            return make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            )
        if texture_type.startswith("texture2d_ms<"):
            return make_descriptor(
                "int2", (("get_width", False), ("get_height", False))
            )
        if texture_type.startswith("texture3d<"):
            return make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_depth", False),
                ),
            )
        return make_descriptor("int2", (("get_width", False), ("get_height", False)))

    def sampled_texture_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        make_descriptor = resource_query_method_size_descriptor
        descriptors = {
            "texture1d<float>": make_descriptor("int", (("get_width", True),)),
            "texture1d_array<float>": make_descriptor(
                "int2", (("get_width", True), ("get_array_size", False))
            ),
            "texture2d<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "depth2d<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "texturecube<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "depthcube<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "texture2d_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "depth2d_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "texturecube_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "depthcube_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "texture3d<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_depth", True),
                ),
            ),
            "texture2d_ms<float>": make_descriptor(
                "int2", (("get_width", False), ("get_height", False))
            ),
            "texture2d_ms_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            ),
        }
        return descriptors.get(texture_type)

    def texture_query_levels_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        if descriptor["multisample"]:
            return texture_query_levels_multisample_expression()
        return f"int({texture_name}.get_num_mip_levels())"

    def texture_samples_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        if not descriptor["multisample"]:
            return unsupported_texture_samples_query_call_expression("Metal")
        return texture_samples_query_expression("Metal", texture_name)

    def image_coordinate_expression(self, image_type, coord):
        image_type = self.storage_image_access_agnostic_type(image_type)
        if image_type in {
            "texture1d_array<float, access::read_write>",
            "texture1d_array<int, access::read_write>",
            "texture1d_array<uint, access::read_write>",
        }:
            coord_x = self.unsigned_coordinate_expression(
                self.vector_component(coord, "x"), 1
            )
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer
        if image_type in {
            "texture1d<float, access::read_write>",
            "texture1d<int, access::read_write>",
            "texture1d<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 1), None
        if image_type in {
            "texture2d_array<float, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d_array<uint, access::read_write>",
            "texture2d_ms_array<float, access::read_write>",
            "texture2d_ms_array<int, access::read_write>",
            "texture2d_ms_array<uint, access::read_write>",
        }:
            coord_xy = self.unsigned_coordinate_expression(
                self.vector_component(coord, "xy"), 2
            )
            layer = f"uint({self.vector_component(coord, 'z')})"
            return coord_xy, layer
        if image_type in {
            "texture2d_ms<float, access::read_write>",
            "texture2d_ms<int, access::read_write>",
            "texture2d_ms<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 2), None
        if image_type in {
            "texture3d<float, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture3d<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 3), None
        return self.unsigned_coordinate_expression(coord, 2), None

    def unsigned_coordinate_expression(self, coord, dimensions):
        constructor = "uint" if dimensions == 1 else f"uint{dimensions}"
        coord_text = str(coord).strip()
        if coord_text.startswith(f"{constructor}("):
            return coord_text
        return f"{constructor}({coord_text})"

    def storage_image_access_agnostic_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        return metal_storage_image_access_agnostic_type(image_type)

    def is_integer_image_type(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        return is_metal_integer_image_type(image_type)

    def is_scalar_image_format(self, image_format):
        return is_scalar_image_format(image_format)

    def is_two_component_image_format(self, image_format):
        return is_two_component_image_format(image_format)

    def is_scalar_integer_image_resource(self, image_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(image_type)

    def is_float_image_resource(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        return is_metal_float_image_resource(image_type)

    def image_store_constructors_by_kind(self):
        return storage_image_store_constructors("float4", "int4", "uint4")

    def image_store_zero_values_by_kind(self):
        return storage_image_zero_values()

    def image_load_component_suffix(self, image_type, image_format):
        if self.suppress_image_load_component_suffix:
            return ""
        return storage_image_load_component_suffix(
            image_format,
            expected_scalar=self.is_scalar_value_type(
                self.current_expression_expected_type
            ),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                image_type, image_format
            )
            and self.expected_component_count() != 4,
            float_resource=self.is_float_image_resource(image_type),
        )

    def image_format_store_constructor(self, image_format):
        return storage_image_format_store_constructor(
            image_format, self.image_store_constructors_by_kind()
        )

    def integer_image_store_constructor(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        if image_type in {
            "texture1d<int, access::read_write>",
            "texture1d_array<int, access::read_write>",
            "texture2d<int, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d_ms<int, access::read_write>",
            "texture2d_ms_array<int, access::read_write>",
        }:
            return "int4"
        if image_type in {
            "texture1d<uint, access::read_write>",
            "texture1d_array<uint, access::read_write>",
            "texture2d<uint, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<uint, access::read_write>",
            "texture2d_ms<uint, access::read_write>",
            "texture2d_ms_array<uint, access::read_write>",
        }:
            return "uint4"
        return None

    def two_component_image_store_expression(
        self, image_format, value, value_type=None
    ):
        return storage_image_two_component_store_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            self.image_store_constructors_by_kind(),
            self.image_store_zero_values_by_kind(),
        )

    def image_store_value_expression(
        self, image_type, image_format, value, value_type=None
    ):
        scalar_integer_resource = self.is_scalar_integer_image_resource(
            image_type, image_format
        )
        if self.is_vector_value_type(value_type):
            scalar_integer_resource = False
        return storage_image_store_value_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            scalar_integer_resource=scalar_integer_resource,
            float_resource=self.is_float_image_resource(image_type),
            integer_constructor=self.integer_image_store_constructor(image_type),
            float_constructor="float4",
            constructors_by_kind=self.image_store_constructors_by_kind(),
            zero_values_by_kind=self.image_store_zero_values_by_kind(),
        )

    def image_store_expected_value_type(self, image_type, image_format, value_arg=None):
        if value_arg is not None and self.is_vector_value_type(
            self.expression_result_type(value_arg)
        ):
            return None
        if image_format:
            return image_format_component_kind(image_format)
        component_type = metal_storage_image_component_type(image_type)
        if component_type in {"float", "int", "uint"}:
            return component_type
        return None

    def image_load_result_type(self, image_arg):
        image_format = self.image_resource_format(image_arg)
        if image_format:
            return image_format_result_type(image_format)
        image_type = self.texture_argument_resource_type(image_arg)
        return metal_storage_image_component_type(image_type)

    def image_load_component_kind(self, image_type, image_format):
        if image_format:
            return image_format_component_kind(image_format)
        component_type = metal_storage_image_component_type(image_type)
        if component_type in {"float", "int", "uint"}:
            return component_type
        return None

    def image_load_channel_count(self, image_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                metal_storage_image_component_type(image_type)
            ),
        )

    def expected_component_kind(self):
        return numeric_component_kind_from_type(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def expected_component_count(self):
        return self.value_component_count(self.current_expression_expected_type)

    def expression_component_kind(self, expr):
        return numeric_expression_component_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def value_component_count(self, vtype):
        return numeric_component_count_from_type(
            vtype,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={"float", "half", "double", "int", "uint", "bool"},
            excluded_type_markers=("x",),
        )

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={"float", "half", "double", "int", "uint", "bool"},
            excluded_type_markers=("x",),
        )

    def image_store_channel_count(self, image_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                metal_storage_image_component_type(image_type)
            ),
        )

    def image_format_allows_native_vector_context(self, image_format):
        return image_format in {"r32f", "r32i", "r32ui"}

    def image_format_context_channel_count_allowed(
        self, image_format, format_channels, context_channels
    ):
        if format_channels == context_channels:
            return True
        if not self.image_format_allows_native_vector_context(image_format):
            return False
        return format_channels == 1 and context_channels in {1, 4}

    def validate_image_store_value_shape(self, image_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(image_type, image_format)
        value_channels = self.expression_component_count(value_arg)
        if self.image_format_context_channel_count_allowed(
            image_format, expected_channels, value_channels
        ):
            return
        value_channels = image_store_value_shape_mismatch(
            expected_channels, value_channels
        )
        if value_channels is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_store_value_shape_error(
                "Metal",
                format_label,
                expression_debug_name(value_arg),
                expected_channels,
                value_channels,
            )
        )

    def validate_image_store_value_type(self, image_type, image_format, value_arg):
        self.validate_image_store_value_shape(image_type, image_format, value_arg)
        expected_kind = self.image_store_expected_value_type(image_type, image_format)
        value_kind = image_store_value_kind_mismatch(
            expected_kind, self.expression_component_kind(value_arg)
        )
        if value_kind is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_store_value_kind_error(
                "Metal",
                format_label,
                expression_debug_name(value_arg),
                expected_kind,
                value_kind,
            )
        )

    def validate_image_load_result_type(self, image_type, image_format):
        expected_kind = self.expected_component_kind()
        component_kind = self.image_load_component_kind(image_type, image_format)
        format_label = image_format or self.resource_base_type(image_type)
        if not should_validate_image_load_result_shape(expected_kind, component_kind):
            return
        expected_kind = image_load_result_kind_mismatch(expected_kind, component_kind)
        if expected_kind is not None:
            raise ValueError(
                image_load_result_kind_error(
                    "Metal", format_label, component_kind, expected_kind
                )
            )
        expected_channels = self.expected_component_count()
        loaded_channels = self.image_load_channel_count(image_type, image_format)
        if self.image_format_context_channel_count_allowed(
            image_format, loaded_channels, expected_channels
        ):
            return
        expected_channels = image_load_result_shape_mismatch(
            loaded_channels,
            expected_channels,
        )
        if expected_channels is None:
            return
        raise ValueError(
            image_load_result_shape_error(
                "Metal", format_label, loaded_channels, expected_channels
            )
        )

    def image_atomic_method(self, func_name):
        return {
            "imageAtomicAdd": "atomic_fetch_add",
            "imageAtomicMin": "atomic_fetch_min",
            "imageAtomicMax": "atomic_fetch_max",
            "imageAtomicAnd": "atomic_fetch_and",
            "imageAtomicOr": "atomic_fetch_or",
            "imageAtomicXor": "atomic_fetch_xor",
            "imageAtomicExchange": "atomic_exchange",
        }.get(func_name)

    def unsupported_multisample_image_atomic_call(self, func_name, image_type):
        resource_type = self.resource_base_type(image_type)
        agnostic_type = self.storage_image_access_agnostic_type(image_type)
        component_type = metal_storage_image_component_type(agnostic_type)
        return unsupported_multisample_image_atomic_expression(
            "Metal",
            func_name,
            resource_type,
            storage_image_atomic_zero_value(component_type),
        )

    def unsupported_multisample_image_store_call(self, image_type):
        return unsupported_multisample_image_store_expression(
            "Metal", self.resource_base_type(image_type)
        )

    def unsupported_image_atomic_call(self, func_name, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        component_type = metal_storage_image_component_type(image_type)
        return unsupported_image_atomic_expression(
            "Metal",
            func_name,
            self.resource_base_type(image_type),
            storage_image_atomic_zero_value(component_type),
        )

    def image_atomic_compare_descriptor(self, texture_type):
        texture_type = self.storage_image_access_agnostic_type(texture_type)
        component_type = metal_storage_image_component_type(texture_type)
        if "<" not in texture_type:
            return None

        texture_family = texture_type.split("<", 1)[0]
        metadata = image_atomic_helper_resource_metadata(
            texture_family,
            {
                "texture1d": "image1D",
                "texture1d_array": "image1DArray",
                "texture2d": "image2D",
                "texture3d": "image3D",
                "texture2d_array": "image2DArray",
            },
            {
                "texture1d": "int",
                "texture1d_array": "int2",
                "texture2d": "int2",
                "texture3d": "int3",
                "texture2d_array": "int3",
            },
            extra_fields_by_family={
                "texture1d": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint(coord), &original, value)"
                    )
                },
                "texture1d_array": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint(coord.x), uint(coord.y), &original, value)"
                    )
                },
                "texture2d": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint2(coord), &original, value)"
                    )
                },
                "texture3d": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint3(coord), &original, value)"
                    )
                },
                "texture2d_array": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
                    )
                },
            },
        )
        if metadata is None:
            return None

        descriptor = image_atomic_helper_descriptor_fields(
            "imageAtomicCompSwap",
            component_type,
            metadata["suffix_family"],
            metadata["coord_type"],
        )
        if descriptor is None:
            return None
        descriptor.update(
            {
                "vector_type": f"{component_type}4",
                "exchange_expr": metadata["exchange_expr"],
            }
        )
        return descriptor

    def image_atomic_compare_helper_name(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["helper_name"] if descriptor else None

    def image_atomic_compare_return_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["return_type"] if descriptor else None

    def image_atomic_compare_vector_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["vector_type"] if descriptor else None

    def image_atomic_compare_coord_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["coord_type"] if descriptor else None

    def image_atomic_compare_exchange_expression(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["exchange_expr"] if descriptor else None

    def generate_image_atomic_compare_helpers(self):
        if not self.required_image_atomic_compare_helpers:
            return ""

        helpers = []
        for texture_type in sorted(self.required_image_atomic_compare_helpers):
            helper_name = self.image_atomic_compare_helper_name(texture_type)
            return_type = self.image_atomic_compare_return_type(texture_type)
            vector_type = self.image_atomic_compare_vector_type(texture_type)
            coord_type = self.image_atomic_compare_coord_type(texture_type)
            exchange_expr = self.image_atomic_compare_exchange_expression(texture_type)
            if (
                not helper_name
                or not return_type
                or not vector_type
                or not coord_type
                or not exchange_expr
            ):
                continue
            helpers.append(
                f"{return_type} {helper_name}({texture_type} image, {coord_type} coord, {return_type} compareValue, {return_type} value) {{\n"
                f"    {vector_type} original;\n"
                "    do {\n"
                "        original.x = compareValue;\n"
                f"    }} while (!{exchange_expr} && original.x == compareValue);\n"
                "    return original.x;\n"
                "}\n\n"
            )
        return "".join(helpers)

    def buffer_atomic_compare_helper_name(self, component_type):
        return f"__crossgl_buffer_atomic_compare_exchange_{component_type}"

    def generate_buffer_atomic_compare_helpers(self):
        if not self.required_buffer_atomic_compare_helpers:
            return ""

        helpers = []
        for component_type in sorted(self.required_buffer_atomic_compare_helpers):
            value_type = self.map_type(component_type)
            atomic_type = f"atomic_{component_type}"
            helper_name = self.buffer_atomic_compare_helper_name(component_type)
            helpers.append(
                f"{value_type} {helper_name}(device uchar* buffer, uint offset, {value_type} compareValue, {value_type} value) {{\n"
                f"    device {atomic_type}* target = reinterpret_cast<device {atomic_type}*>(buffer + offset);\n"
                f"    {value_type} original;\n"
                "    do {\n"
                "        original = compareValue;\n"
                "    } while (!atomic_compare_exchange_weak_explicit(target, &original, value, memory_order_relaxed, memory_order_relaxed) && original == compareValue);\n"
                "    return original;\n"
                "}\n\n"
            )
        return "".join(helpers)

    def generate_image_call(self, func_name, args):
        if func_name == "imageAtomicCompSwap" and len(args) >= 4:
            image_type = self.texture_resource_type(args[0])
            if self.is_multisample_storage_image_resource(image_type):
                return self.unsupported_multisample_image_atomic_call(
                    func_name, image_type
                )
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            helper_name = self.image_atomic_compare_helper_name(image_type)
            if not helper_name:
                return self.unsupported_image_atomic_call(func_name, image_type)
            self.required_image_atomic_compare_helpers.add(image_type)
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"

        atomic_method = self.image_atomic_method(func_name)
        if atomic_method and len(args) >= 3:
            image_type = self.texture_resource_type(args[0])
            if self.is_multisample_storage_image_resource(image_type):
                return self.unsupported_multisample_image_atomic_call(
                    func_name, image_type
                )
            if metal_storage_image_component_type(image_type) == "float":
                return self.unsupported_image_atomic_call(func_name, image_type)
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if layer is not None:
                return (
                    f"{image_name}.{atomic_method}({texel_coord}, {layer}, {value}).x"
                )
            return f"{image_name}.{atomic_method}({texel_coord}, {value}).x"

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            image_type = self.texture_resource_type(args[0])
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if len(args) >= 3 and self.is_multisample_storage_image_resource(
                image_type
            ):
                sample = f"uint({self.generate_expression(args[2])})"
                if layer is not None:
                    load_expr = f"{image_name}.read({texel_coord}, {layer}, {sample})"
                else:
                    load_expr = f"{image_name}.read({texel_coord}, {sample})"
            elif layer is not None:
                load_expr = f"{image_name}.read({texel_coord}, {layer})"
            else:
                load_expr = f"{image_name}.read({texel_coord})"
            image_format = self.image_resource_format(args[0])
            self.validate_image_load_result_type(image_type, image_format)
            return f"{load_expr}{self.image_load_component_suffix(image_type, image_format)}"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            image_type = self.texture_resource_type(args[0])
            if len(args) >= 4 and self.is_multisample_storage_image_resource(
                image_type
            ):
                sample = f"uint({self.generate_expression(args[2])})"
                value_arg = args[3]
            else:
                sample = None
                value_arg = args[2]
            image_format = self.image_resource_format(args[0])
            self.validate_image_store_value_type(image_type, image_format, value_arg)
            value = self.generate_expression_with_expected(
                value_arg,
                self.image_store_expected_value_type(
                    image_type, image_format, value_arg
                ),
            )
            value = self.image_store_value_expression(
                image_type, image_format, value, self.expression_result_type(value_arg)
            )
            if self.is_multisample_storage_image_resource(image_type):
                return self.unsupported_multisample_image_store_call(image_type)
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if sample is not None and layer is not None:
                return f"{image_name}.write({value}, {texel_coord}, {layer}, {sample})"
            if sample is not None:
                return f"{image_name}.write({value}, {texel_coord}, {sample})"
            if layer is not None:
                return f"{image_name}.write({value}, {texel_coord}, {layer})"
            return f"{image_name}.write({value}, {texel_coord})"

        return None

    def generate_texture_call(self, func_name, args):
        if not func_name:
            return None

        self.validate_texture_call_arity(func_name, args)
        self.validate_image_resource_argument(func_name, args)
        self.validate_image_access_argument(func_name, args)
        self.validate_texture_resource_argument(func_name, args)
        self.validate_image_multisample_arguments(func_name, args)
        self.validate_image_atomic_format_argument(func_name, args)
        self.validate_integer_coordinate_argument(func_name, args)
        self.validate_coordinate_dimension_argument(func_name, args)
        self.validate_query_lod_coordinate_argument(func_name, args)
        self.validate_compare_argument(func_name, args)
        self.validate_lod_argument(func_name, args)
        self.validate_bias_argument(func_name, args)
        self.validate_sample_index_argument(func_name, args)
        self.validate_mip_level_argument(func_name, args)
        self.validate_gradient_dimension_arguments(func_name, args)
        self.validate_offset_dimension_argument(func_name, args)
        self.validate_gather_component_argument(func_name, args)

        image_call = self.generate_image_call(func_name, args)
        if image_call is not None:
            return image_call

        if is_resource_size_query_operation(func_name) and args:
            lod_arg = args[1] if len(args) > 1 else None
            return self.texture_query_size_expression(args[0], lod_arg)

        if is_texture_query_levels_operation(func_name) and args:
            return self.texture_query_levels_expression(args[0])

        if is_resource_samples_query_operation(func_name) and args:
            return self.texture_samples_expression(args[0])

        if len(args) < 2:
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None

        texture_name, sampler_arg, coord, extra_args = parts
        texture_type = self.texture_argument_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        packed_shadow_call = self.generate_packed_shadow_texture_sample_call(
            func_name,
            texture_name,
            sampler_arg,
            coord,
            extra_args,
            texture_type,
        )
        if packed_shadow_call is not None:
            return packed_shadow_call

        is_array_texture = self.is_array_texture_resource(texture_type)
        if is_array_texture:
            coord_xy, layer = self.texture_coordinate_parts(texture_type, coord)

        if is_texture_sample_operation(
            func_name
        ) and self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)

        if is_texture_sample_basic_operation(func_name):
            if extra_args:
                if self.is_texture1d_sample_resource(texture_type):
                    return self.unsupported_texture1d_sampling_option_call(
                        func_name, texture_type
                    )
                bias = self.generate_expression(extra_args[0])
                if is_array_texture:
                    return (
                        f"{texture_name}.sample("
                        f"{sampler_arg}, {coord_xy}, {layer}, bias({bias}))"
                    )
                return f"{texture_name}.sample({sampler_arg}, {coord}, bias({bias}))"
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer})"
            return f"{texture_name}.sample({sampler_arg}, {coord})"
        if is_texture_sample_lod_operation(func_name) and extra_args:
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture1d_sampling_option_call(
                    func_name, texture_type
                )
            lod = self.generate_expression(extra_args[0])
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer}, level({lod}))"
            return f"{texture_name}.sample({sampler_arg}, {coord}, level({lod}))"
        if is_texture_sample_grad_operation(func_name) and len(extra_args) >= 2:
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture1d_sampling_option_call(
                    func_name, texture_type
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer}, {gradient_options})"
            return f"{texture_name}.sample({sampler_arg}, {coord}, {gradient_options})"
        if is_texture_sample_offset_operation(func_name):
            return self.generate_texture_sample_offset_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
            )
        if is_projected_texture_operation(func_name):
            return self.generate_texture_projected_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
                args,
            )
        if is_texture_gather_operation(func_name):
            return self.generate_texture_gather_call(
                func_name, texture_name, sampler_arg, coord, extra_args, texture_type
            )
        if is_texture_compare_operation(func_name):
            return self.generate_texture_compare_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
                args,
            )
        if is_texture_gather_compare_operation(func_name):
            return self.generate_texture_gather_compare_call(
                func_name, texture_name, sampler_arg, coord, extra_args, texture_type
            )
        if is_texture_query_lod_operation(func_name):
            if self.is_multisample_texture_resource(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_resource(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            lod_coord = self.texture_query_lod_coordinate(texture_type, coord)
            return (
                f"float2({texture_name}.calculate_clamped_lod({sampler_arg}, {lod_coord}), "
                f"{texture_name}.calculate_unclamped_lod({sampler_arg}, {lod_coord}))"
            )
        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            texel_coord, layer = self.texture_read_coordinate_parts(texture_type, coord)
            if self.is_multisample_texture_resource(texture_type):
                sample = self.unsigned_coordinate_expression(
                    self.generate_expression(args[2]), 1
                )
                if layer is not None:
                    return f"{texture_name}.read({texel_coord}, {layer}, {sample})"
                return f"{texture_name}.read({texel_coord}, {sample})"
            if self.is_texture1d_sample_resource(texture_type):
                lod = self.metal_texture1d_read_lod_argument(args[2])
                if lod is None:
                    return self.unsupported_texture1d_texel_fetch_lod_call(
                        func_name, texture_type
                    )
                if layer is not None:
                    return f"{texture_name}.read({texel_coord}, {layer}, {lod})"
                return f"{texture_name}.read({texel_coord}, {lod})"
            lod = self.unsigned_coordinate_expression(
                self.generate_expression(args[2]), 1
            )
            if layer is not None:
                return f"{texture_name}.read({texel_coord}, {layer}, {lod})"
            return f"{texture_name}.read({texel_coord}, {lod})"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource(texture_type):
                return unsupported_multisample_texel_fetch_offset_expression("Metal")
            if self.is_texture1d_sample_resource(texture_type):
                lod = self.metal_texture1d_read_lod_argument(args[2])
                if lod is None:
                    return self.unsupported_texture1d_texel_fetch_lod_call(
                        func_name, texture_type
                    )
                offset = self.generate_expression(args[3])
                if is_array_texture:
                    texel_coord, layer = self.texture_coordinate_parts(
                        texture_type, coord
                    )
                    return (
                        f"{texture_name}.read(uint(({texel_coord} + {offset})), "
                        f"{layer}, {lod})"
                    )
                return f"{texture_name}.read(uint(({coord} + {offset})), {lod})"
            lod = self.unsigned_coordinate_expression(
                self.generate_expression(args[2]), 1
            )
            offset = self.generate_expression(args[3])
            (
                texel_coord,
                layer,
                dimensions,
            ) = self.texture_read_coordinate_components(texture_type, coord)
            if layer is not None:
                offset_coord = self.unsigned_coordinate_expression(
                    f"({texel_coord} + {offset})", dimensions
                )
                return f"{texture_name}.read({offset_coord}, {layer}, {lod})"
            offset_coord = self.unsigned_coordinate_expression(
                f"({texel_coord} + {offset})", dimensions
            )
            return f"{texture_name}.read({offset_coord}, {lod})"

        return None

    def generate_texture_min_lod_clamp_call(self, func_name, args):
        if func_name not in {"textureMinLodClamp", "textureMinLodClampOffset"}:
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None

        texture_name, sampler_arg, coord, extra_args = parts
        expected_extra_args = 2 if func_name == "textureMinLodClampOffset" else 1
        if len(extra_args) != expected_extra_args:
            return None

        texture_type = self.texture_argument_resource_type(args[0])
        sample_args = [sampler_arg]
        if self.is_array_texture_resource(texture_type):
            coord_xy, layer = self.texture_coordinate_parts(texture_type, coord)
            sample_args.extend([coord_xy, layer])
        else:
            sample_args.append(coord)

        min_lod = self.generate_expression(extra_args[0])
        sample_args.append(f"min_lod_clamp({min_lod})")
        if expected_extra_args == 2:
            sample_args.append(self.generate_expression(extra_args[1]))

        return f"{texture_name}.sample({', '.join(sample_args)})"

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if isinstance(type_node, FunctionCallNode):
            return self.safe_expression_to_string(type_node)
        if isinstance(type_node, IdentifierNode):
            return type_node.name
        if isinstance(type_node, PointerType):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if isinstance(type_node, ReferenceType):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            return f"{referenced_type}&"
        generic_args = getattr(type_node, "generic_args", [])
        if hasattr(type_node, "name") and generic_args:
            args = ", ".join(
                self.convert_type_node_to_string(arg)
                or self.metal_tessellation_patch_argument_text(arg)
                for arg in generic_args
            )
            return f"{type_node.name}<{args}>"
        if hasattr(type_node, "name"):
            return type_node.name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.rows == type_node.cols:
                return f"float{type_node.rows}x{type_node.rows}"
            else:
                return f"float{type_node.cols}x{type_node.rows}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return f"{element_type}[{type_node.size}]"
                    else:
                        size_str = self.safe_expression_to_string(type_node.size)
                        return f"{element_type}[{size_str}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "int":
                    return f"int{size}"
                elif element_type == "uint":
                    return f"uint{size}"
                elif element_type == "bool":
                    return f"bool{size}"
                elif element_type in {
                    "f16",
                    "f32",
                    "f64",
                    "i8",
                    "u8",
                    "i16",
                    "u16",
                    "i32",
                    "u32",
                }:
                    return f"vec{size}<{element_type}>"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

    def safe_expression_to_string(self, expr):
        """Convert an expression node to a string representation safely (avoid infinite recursion)."""
        return self.safe_expression_to_string_with_precedence(expr)

    def safe_expression_to_string_with_precedence(self, expr, parent_precedence=0):
        if hasattr(expr, "value"):
            return str(expr.value)
        elif isinstance(expr, FunctionCallNode):
            callee = self.safe_expression_to_string_with_precedence(expr.function)
            generic_args = getattr(expr, "generic_args", []) or []
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                callee = f"{callee}<{args}>"
            arguments = ", ".join(
                self.safe_expression_to_string_with_precedence(argument)
                for argument in getattr(expr, "arguments", [])
            )
            return f"{callee}({arguments})"
        elif isinstance(expr, MemberAccessNode):
            object_expr = self.safe_expression_to_string_with_precedence(
                expr.object_expr
            )
            return f"{object_expr}.{expr.member}"
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, BinaryOpNode):
            operator = self.map_operator(expr.op)
            precedence = self.expression_precedence(operator)
            left = self.safe_expression_to_string_with_precedence(expr.left, precedence)
            right = self.safe_expression_to_string_with_precedence(
                expr.right, precedence + 1
            )
            expression = f"{left} {operator} {right}"
            if precedence < parent_precedence:
                return f"({expression})"
            return expression
        elif isinstance(expr, UnaryOpNode):
            operand = self.safe_expression_to_string_with_precedence(
                expr.operand, self.expression_precedence("unary")
            )
            return f"{self.map_operator(expr.op)}{operand}"
        else:
            # Fallback - avoid calling generate_expression to prevent infinite recursion
            return str(expr)

    def expression_precedence(self, operator):
        return {
            "||": 1,
            "&&": 2,
            "|": 3,
            "^": 4,
            "&": 5,
            "==": 6,
            "!=": 6,
            "<": 7,
            ">": 7,
            "<=": 7,
            ">=": 7,
            "<<": 8,
            ">>": 8,
            "+": 9,
            "-": 9,
            "*": 10,
            "/": 10,
            "%": 10,
            "unary": 11,
        }.get(operator, 0)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        return self.safe_expression_to_string(expr)

    def map_type(self, vtype):
        """Map types to Metal equivalents, handling both strings and TypeNode objects."""
        if vtype is None:
            return "float"

        cooperative_matrix = self.cooperative_matrix_contract(vtype)
        if cooperative_matrix is not None:
            return self.map_cooperative_matrix_type(vtype, cooperative_matrix)
        if isinstance(vtype, PointerType):
            return f"{self.map_type(vtype.pointee_type)}*"
        if isinstance(vtype, ReferenceType):
            return f"{self.map_type(vtype.referenced_type)}&"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        tessellation_patch_type = self.metal_tessellation_patch_mapped_type(vtype)
        if tessellation_patch_type is not None:
            return tessellation_patch_type

        geometry_stream_type = self.metal_geometry_stream_mapped_type(vtype)
        if geometry_stream_type is not None:
            return geometry_stream_type

        if self.is_metal_ray_query_type_name(vtype_str):
            self.require_metal_ray_query_runtime()
            return "CglRayQuery"

        if self.requires_metal_builtin_ray_desc(vtype_str):
            self.required_metal_ray_desc_runtime = True
            return "CglRayDesc"

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_type(base_type)
            return f"{base_mapped}{array_suffix}"

        if self.is_visible_function_table_type(vtype_str):
            return self.map_visible_function_table_type(vtype_str)

        if self.is_intersection_function_table_type(vtype_str):
            return self.map_intersection_function_table_type(vtype_str)

        generic_enum_type = generic_enum_specialized_type_name(self, vtype_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_struct_type = generic_struct_specialized_type_name(self, vtype_str)
        if generic_struct_type is not None:
            return generic_struct_type

        option_payload = self.lowerable_option_payload_type_name(vtype_str)
        if option_payload is not None:
            return self.map_type(option_payload)

        if vtype_str in getattr(self, "enum_type_names", set()):
            return "int"

        if vtype_str in getattr(self, "enum_struct_type_names", set()):
            return vtype_str

        return self.type_mapping.get(vtype_str, vtype_str)

    def cooperative_matrix_contract(self, matrix_type):
        if isinstance(matrix_type, CooperativeMatrixType):
            return (
                matrix_type.element_type,
                matrix_type.rows,
                matrix_type.cols,
                matrix_type.scope,
                matrix_type.use,
                matrix_type.layout,
            )

        type_name = str(matrix_type).strip()
        base_name, generic_args = generic_type_parts(type_name)
        if base_name.rsplit("::", 1)[-1] != "CooperativeMatrix" or not (
            3 <= len(generic_args) <= 6
        ):
            return None
        defaults = ("subgroup", "unspecified", "unspecified")
        generic_args = [*generic_args, *defaults[len(generic_args) - 3 :]]
        return tuple(generic_args)

    def map_cooperative_matrix_type(self, matrix_type, contract):
        """Map the native subset of the canonical cooperative-matrix contract."""
        element_type, rows, cols, scope, use, layout = contract
        metadata = (scope, use, layout)
        if metadata != ("subgroup", "unspecified", "unspecified"):
            raise UnsupportedMetalFeatureError(
                "cooperative-matrix-type-contract",
                "Metal simdgroup_matrix cannot preserve cooperative-matrix "
                "scope/use/layout contract "
                f"{metadata!r}",
                missing_capabilities=("metal.cooperative-matrix-contract-mapping",),
                reason="unsupported-type-contract",
                source_location=getattr(matrix_type, "source_location", None),
            )

        mapped_element_type = self.map_type(element_type)
        rows_text = self.convert_type_node_to_string(rows)
        cols_text = self.convert_type_node_to_string(cols)
        return f"simdgroup_matrix<{mapped_element_type}, {rows_text}, {cols_text}>"

    def metal_geometry_stream_info(self, vtype):
        if vtype is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if isinstance(vtype, (PointerType, ReferenceType)):
            return None

        name = getattr(vtype, "name", None)
        generic_args = getattr(vtype, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.map_type(generic_args[0])

        type_name = self.type_name_string(vtype)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return None

        base_name, generic_arg = type_name.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None
        generic_arg = generic_arg[:-1].strip()
        if not generic_arg:
            return None
        return base_name, self.map_type(generic_arg)

    def metal_geometry_stream_mapped_type(self, vtype):
        stream_info = self.metal_geometry_stream_info(vtype)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CrossGLMetal{stream_name}<{output_type}>"

    def format_metal_geometry_stream_parameter(self, raw_param_type, name):
        stream_type = self.metal_geometry_stream_mapped_type(raw_param_type)
        if stream_type is None:
            return None
        return f"thread {stream_type}& {name}"

    def split_metal_generic_argument_string(self, arguments):
        split_arguments = []
        start = 0
        depth = 0
        for index, char in enumerate(arguments):
            if char == "<":
                depth += 1
            elif char == ">":
                depth = max(depth - 1, 0)
            elif char == "," and depth == 0:
                split_arguments.append(arguments[start:index].strip())
                start = index + 1
        split_arguments.append(arguments[start:].strip())
        return [argument for argument in split_arguments if argument]

    def metal_type_generic_arguments(self, vtype):
        generic_args = getattr(vtype, "generic_args", None)
        if generic_args:
            return generic_args

        type_name = self.type_name_string(vtype)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return []
        arguments = type_name.split("<", 1)[1][:-1].strip()
        return self.split_metal_generic_argument_string(arguments)

    def metal_tessellation_patch_type_base(self, vtype):
        if isinstance(vtype, PointerType):
            vtype = vtype.pointee_type
        elif isinstance(vtype, ReferenceType):
            vtype = vtype.referenced_type

        name = getattr(vtype, "name", None)
        if name:
            return name

        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        return type_name.split("<", 1)[0].strip()

    def metal_tessellation_patch_argument_text(self, argument):
        if argument is None:
            return None
        if isinstance(argument, str):
            return argument
        if hasattr(argument, "value"):
            return str(argument.value)
        if hasattr(argument, "name"):
            return str(argument.name)
        return self.type_name_string(argument) or str(argument)

    def metal_tessellation_patch_info(self, vtype):
        if isinstance(vtype, PointerType):
            vtype = vtype.pointee_type
        elif isinstance(vtype, ReferenceType):
            vtype = vtype.referenced_type

        patch_type = self.metal_tessellation_patch_type_base(vtype)
        if patch_type not in {"InputPatch", "OutputPatch"}:
            return None

        generic_args = self.metal_type_generic_arguments(vtype)
        valid = len(generic_args) == 2
        element_type = None
        count_text = None
        count_value = None
        if valid:
            element_type = self.map_type(generic_args[0])
            count_text = self.metal_tessellation_patch_argument_text(generic_args[1])
            count_value = self.literal_int_value(
                generic_args[1], self.literal_int_constants
            )
            if count_value is None:
                count_value = self.literal_int_value(
                    count_text, self.literal_int_constants
                )

        return {
            "kind": patch_type,
            "valid": valid,
            "element_type": element_type,
            "count_text": count_text,
            "count_value": count_value,
        }

    def metal_tessellation_patch_mapped_type(self, vtype):
        info = self.metal_tessellation_patch_info(vtype)
        if info is None or not info["valid"]:
            return None
        if not info["element_type"] or not info["count_text"]:
            return None
        return (
            f"CrossGLMetal{info['kind']}<"
            f"{info['element_type']}, {info['count_text']}>"
        )

    def format_metal_tessellation_patch_parameter(self, raw_param_type, name):
        patch_type = self.metal_tessellation_patch_mapped_type(raw_param_type)
        if patch_type is None:
            return None
        return f"thread const {patch_type}& {name}"

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_OR": "|=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_XOR": "^=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "LOGICAL_AND": "&&",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to Metal attribute syntax."""
        if semantic is not None:
            hlsl_attribute = self.metal_hlsl_attribute_semantic(semantic)
            if hlsl_attribute is not None:
                return f" [[{hlsl_attribute}]]"
            hlsl_color = self.metal_hlsl_color_semantic(semantic)
            if hlsl_color is not None:
                return f" [[user({hlsl_color})]]"
            mapped_semantic = self.semantic_map.get(str(semantic), str(semantic))
            if (
                self.is_metal_tessellation_helper_semantic(semantic)
                and mapped_semantic != "primitive_id"
            ):
                return ""
            if mapped_semantic.startswith("[[") and mapped_semantic.endswith("]]"):
                return f" {mapped_semantic}"
            else:
                return f" [[{mapped_semantic}]]"
        else:
            return ""

    def metal_hlsl_attribute_semantic(self, semantic):
        match = re.fullmatch(r"ATTRIB(\d+)", str(semantic), re.IGNORECASE)
        if match:
            return f"attribute({match.group(1)})"
        return None

    def metal_hlsl_color_semantic(self, semantic):
        match = re.fullmatch(r"COLOR(\d+)", str(semantic), re.IGNORECASE)
        if not match:
            return None
        return f"Color{match.group(1)}"

    def metal_array_semantic_attribute_precedes_extent(self, semantic):
        return self.canonical_metal_semantic(semantic) in {"clip_distance"}

    def is_metal_tessellation_helper_semantic(self, semantic):
        if semantic is None:
            return False
        return str(semantic).lower() in {
            "sv_domainlocation",
            "sv_insidetessfactor",
            "sv_outputcontrolpointid",
            "sv_primitiveid",
            "sv_tessfactor",
        }
