"""CrossGL-to-GLSL code generator."""

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    ArrayNode,
    ArrayAccessNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    ConstructorNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PreprocessorNode,
    RayQueryOpNode,
    RayTracingOpNode,
    RangeNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from .array_utils import (
    parse_array_type,
    format_c_style_array_declaration,
    split_array_type_suffix,
    get_array_size_from_node,
    evaluate_literal_int_expression,
    collect_literal_int_constants,
    collect_struct_member_types,
)
from ..validation import (
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_member_names,
    collect_duplicate_cbuffer_names,
    collect_non_resource_global_resource_shadows,
    expression_debug_name,
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    INTEGER_COORDINATE_INTRINSIC_NAMES,
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
from .stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    collect_stage_local_structs,
    collect_stage_local_variables,
    compute_local_size,
    deduplicate_named_declarations,
    normalize_stage_name,
    order_functions_by_dependencies,
    should_emit_qualified_function,
    stage_matches,
)
from .resource_arrays import collect_resource_array_size_hints
from .enum_utils import (
    collect_enum_type_names,
    collect_enum_struct_variant_fields,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_enum_variant_constants,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_value_expression,
    generate_enum_constructor_expression,
    generate_generic_enum_constructor_functions,
    generate_generic_enum_constants,
    generate_generic_enum_structs,
    generate_enum_constructor_call,
    generate_enum_constructor_functions,
    generate_enum_constants,
    generate_enum_structs,
    generic_enum_specialized_type_name,
    infer_enum_constructor_type,
    sanitize_type_name,
)
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    generate_generic_structs,
    generate_struct_constructor_expression,
    generic_struct_specialized_type_name,
    infer_struct_constructor_type,
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
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    default_storage_image_channel_count,
    explicit_image_format,
    floating_coordinate_dimension_from_type_name,
    image_atomic_explicit_format_component_kind,
    image_atomic_format_error,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
    image_atomic_value_arguments as shared_image_atomic_value_arguments,
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_load_result_shape_mismatch,
    image_format_or_default_channel_count,
    image_format_component_kind,
    image_format_result_type,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    image_store_value_shape_mismatch,
    is_image_format_attribute,
    is_image_atomic_operation,
    is_image_resource_operation,
    integer_coordinate_dimension_from_type_name,
    is_glsl_float_image_resource,
    is_glsl_integer_image_type,
    is_glsl_storage_image_type,
    is_floating_scalar_type_name,
    is_integer_coordinate_type_name,
    is_integer_scalar_type_name,
    is_numeric_scalar_type_name,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_resource_access_attribute,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_basic_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_grad_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_lod_operation,
    is_projected_texture_operation,
    is_scalar_image_format,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_operation,
    is_texture_compare_basic_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_offset_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_offset_operation,
    is_texture_gather_single_offset_operation,
    is_texture_query_levels_operation,
    is_texture_query_lod_operation,
    is_texture_samples_query_operation,
    is_texture_sampling_operation,
    is_texture_sample_offset_operation,
    is_two_component_image_format,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    image_resource_metadata,
    projected_texture_extra_argument_count_error,
    record_explicit_image_metadata,
    should_validate_image_load_result_shape,
    storage_image_atomic_zero_value,
    storage_image_format_store_constructor,
    storage_image_load_component_suffix,
    storage_image_store_constructors,
    storage_image_store_value_expression,
    storage_image_two_component_store_expression,
    storage_image_zero_values,
    supported_image_formats,
    texture_argument_diagnostic_type as shared_texture_argument_diagnostic_type,
    texture_compare_argument_error,
    texture_compare_coordinate_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_lod_array_error,
    texture_compare_projected_coordinate_error,
    texture_coordinate_arguments_error,
    texture_gather_capability_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    texture_image_resource_operation_names,
    texture_query_levels_multisample_expression,
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_swizzle,
    texture_query_lod_coordinate_type_error,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
    texture_samples_query_expression,
    texture_sample_offset_capability_error,
    texture_multisample_sample_type_error,
    validate_texture_operation_arity,
    requires_integer_coordinate,
    unsupported_multisample_texture_call_vector_expression,
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_multisample_texel_fetch_offset_expression,
    unsupported_projected_texture_call_expression,
    unsupported_projected_texture_operation_error,
    unsupported_image_atomic_expression as image_atomic_diagnostic_expression,
    unsupported_cube_texel_fetch_expression,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texture_gather_compare_call_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_compare_operation_error,
    unsupported_texture_offset_call_expression,
    unsupported_texture_query_levels_expression,
    unsupported_texture_query_lod_expression,
    unsupported_texture_samples_query_call_expression,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)


class GLSLCodeGen:
    """Emit GLSL source from the shared CrossGL translator AST."""

    MESH_STAGE_NAMES = {"mesh", "task", "amplification", "object"}
    RAY_STAGE_NAMES = {
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
    RAY_QUERY_METHOD_MAP = {
        "Initialize": ("rayQueryInitializeEXT", None),
        "TraceRayInline": ("rayQueryInitializeEXT", None),
        "Proceed": ("rayQueryProceedEXT", None),
        "Abort": ("rayQueryTerminateEXT", None),
        "Terminate": ("rayQueryTerminateEXT", None),
        "GenerateIntersection": ("rayQueryGenerateIntersectionEXT", None),
        "ConfirmIntersection": ("rayQueryConfirmIntersectionEXT", None),
        "RayTMin": ("rayQueryGetRayTMinEXT", None),
        "RayFlags": ("rayQueryGetRayFlagsEXT", None),
        "WorldRayOrigin": ("rayQueryGetWorldRayOriginEXT", None),
        "WorldRayDirection": ("rayQueryGetWorldRayDirectionEXT", None),
        "CandidateType": ("rayQueryGetIntersectionTypeEXT", "false"),
        "CommittedType": ("rayQueryGetIntersectionTypeEXT", "true"),
        "CandidatePrimitiveIndex": (
            "rayQueryGetIntersectionPrimitiveIndexEXT",
            "false",
        ),
        "CommittedPrimitiveIndex": (
            "rayQueryGetIntersectionPrimitiveIndexEXT",
            "true",
        ),
        "CandidateInstanceID": ("rayQueryGetIntersectionInstanceIdEXT", "false"),
        "CommittedInstanceID": ("rayQueryGetIntersectionInstanceIdEXT", "true"),
        "CandidateGeometryIndex": (
            "rayQueryGetIntersectionGeometryIndexEXT",
            "false",
        ),
        "CommittedGeometryIndex": (
            "rayQueryGetIntersectionGeometryIndexEXT",
            "true",
        ),
        "CandidateInstanceCustomIndex": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT",
            "false",
        ),
        "CommittedInstanceCustomIndex": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT",
            "true",
        ),
        "CandidateInstanceShaderBindingTableRecordOffset": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",
            "false",
        ),
        "CommittedInstanceShaderBindingTableRecordOffset": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",
            "true",
        ),
        "CandidateObjectRayOrigin": (
            "rayQueryGetIntersectionObjectRayOriginEXT",
            "false",
        ),
        "CandidateObjectRayDirection": (
            "rayQueryGetIntersectionObjectRayDirectionEXT",
            "false",
        ),
        "CommittedObjectRayOrigin": (
            "rayQueryGetIntersectionObjectRayOriginEXT",
            "true",
        ),
        "CommittedObjectRayDirection": (
            "rayQueryGetIntersectionObjectRayDirectionEXT",
            "true",
        ),
        "CandidateRayT": ("rayQueryGetIntersectionTEXT", "false"),
        "CommittedRayT": ("rayQueryGetIntersectionTEXT", "true"),
        "CandidateTriangleBarycentrics": (
            "rayQueryGetIntersectionBarycentricsEXT",
            "false",
        ),
        "CommittedTriangleBarycentrics": (
            "rayQueryGetIntersectionBarycentricsEXT",
            "true",
        ),
        "CandidateTriangleFrontFace": (
            "rayQueryGetIntersectionFrontFaceEXT",
            "false",
        ),
        "CommittedTriangleFrontFace": (
            "rayQueryGetIntersectionFrontFaceEXT",
            "true",
        ),
        "CandidateTriangleVertexPositions": (
            "rayQueryGetIntersectionTriangleVertexPositionsEXT",
            "false",
        ),
        "CommittedTriangleVertexPositions": (
            "rayQueryGetIntersectionTriangleVertexPositionsEXT",
            "true",
        ),
        "CandidateAABBOpaque": ("rayQueryGetIntersectionCandidateAABBOpaqueEXT", None),
        "CandidateObjectToWorld": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "false",
        ),
        "CommittedObjectToWorld": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "true",
        ),
        "CandidateObjectToWorld3x4": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "false",
        ),
        "CommittedObjectToWorld3x4": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "true",
        ),
        "CandidateWorldToObject": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "false",
        ),
        "CommittedWorldToObject": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "true",
        ),
        "CandidateWorldToObject3x4": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "false",
        ),
        "CommittedWorldToObject3x4": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "true",
        ),
    }
    RAY_QUERY_POSITION_FETCH_METHODS = {
        "CandidateTriangleVertexPositions",
        "CommittedTriangleVertexPositions",
    }
    GLSL_RESERVED_IDENTIFIERS = {
        "active",
    }

    def __init__(self):
        """Initialize GLSL type maps and per-generation stage/resource state."""
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.current_resource_aliases = {}
        self.current_identifier_aliases = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.function_sampler_parameter_indices = {}
        self.function_parameter_names = {}
        self.function_parameter_types = {}
        self.function_parameter_infos = {}
        self.function_return_types = {}
        self.function_image_access_requirements = {}
        self.function_definitions = {}
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        self.emitted_glsl_resource_specialization_names = set()
        self.unsupported_structured_buffer_array_functions = {}
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.current_stage_output_member_map = {}
        self.current_stage_parameter_aliases = {}
        self.current_target_stage = None
        self.stage_io_used_locations = {}
        self.stage_io_declarations = {}
        self.flattened_stage_variables = set()
        self.structs_by_name = {}
        self.vertex_input_struct_names = set()
        self.vertex_output_struct_names = set()
        self.combined_vertex_output_member_names = set()
        self.fragment_input_struct_names = set()
        self.fragment_input_member_names = set()
        self.fragment_output_struct_names = set()
        self.fragment_output_member_name_maps = {}
        self.fragment_output_member_layout_maps = {}
        self.vertex_input_member_names = set()
        self.current_function_return_type = None
        self.current_stage_return_type = None
        self.current_stage_entry_type = None
        self.current_expression_expected_type = None
        self.current_generic_function_substitutions = {}
        self.match_temp_variable_index = 0
        self.local_variable_types = {}
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.struct_member_types = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.structured_buffer_instance_members = {}
        self.structured_buffer_counter_members = {}
        self.structured_buffer_counter_instances = {}
        self.glsl_buffer_block_struct_names = set()
        self.semantic_map = {
            "gl_VertexID": "gl_VertexID",
            "gl_InstanceID": "gl_InstanceID",
            "gl_IsFrontFace": "gl_FrontFacing",
            "gl_PrimitiveID": "gl_PrimitiveID",
            "POSITION": "layout(location = 0)",
            "COLOR": "layout(location = 13)",
            "COLOR0": "layout(location = 13)",
            "COLOR1": "layout(location = 14)",
            "COLOR2": "layout(location = 15)",
            "COLOR3": "layout(location = 16)",
            "COLOR4": "layout(location = 17)",
            "COLOR5": "layout(location = 18)",
            "COLOR6": "layout(location = 19)",
            "COLOR7": "layout(location = 20)",
            "NORMAL": "layout(location = 1)",
            "TANGENT": "layout(location = 2)",
            "BINORMAL": "layout(location = 3)",
            "TEXCOORD": "layout(location = 4)",
            "TEXCOORD0": "layout(location = 5)",
            "TEXCOORD1": "layout(location = 6)",
            "TEXCOORD2": "layout(location = 7)",
            "TEXCOORD3": "layout(location = 8)",
            "TEXCOORD4": "layout(location = 9)",
            "TEXCOORD5": "layout(location = 10)",
            "TEXCOORD6": "layout(location = 11)",
            "TEXCOORD7": "layout(location = 12)",
            # Vertex outputs
            "gl_Position": "gl_Position",
            "gl_PointSize": "gl_PointSize",
            "gl_ClipDistance": "gl_ClipDistance",
            # Fragment outputs
            "gl_FragColor": "layout(location = 0)",
            "gl_FragColor0": "layout(location = 0)",
            "gl_FragColor1": "layout(location = 1)",
            "gl_FragColor2": "layout(location = 2)",
            "gl_FragColor3": "layout(location = 3)",
            "gl_FragColor4": "layout(location = 4)",
            "gl_FragColor5": "layout(location = 5)",
            "gl_FragColor6": "layout(location = 6)",
            "gl_FragColor7": "layout(location = 7)",
            "gl_FragDepth": "gl_FragDepth",
            # Additional fragment inputs
            "gl_FragCoord": "gl_FragCoord",
            "gl_FrontFacing": "gl_FrontFacing",
            "gl_PointCoord": "gl_PointCoord",
            # Compute shader specific
            "gl_GlobalInvocationID": "gl_GlobalInvocationID",
            "gl_LocalInvocationID": "gl_LocalInvocationID",
            "gl_WorkGroupID": "gl_WorkGroupID",
            "gl_LocalInvocationIndex": "gl_LocalInvocationIndex",
            "gl_WorkGroupSize": "gl_WorkGroupSize",
            "gl_NumWorkGroups": "gl_NumWorkGroups",
        }

        self.type_mapping = {
            # Most types are the same in CrossGL and GLSL
            "vec2": "vec2",
            "vec3": "vec3",
            "vec4": "vec4",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "packed_float2": "vec2",
            "packed_float3": "vec3",
            "packed_float4": "vec4",
            "simd_float2": "vec2",
            "simd_float3": "vec3",
            "simd_float4": "vec4",
            "half": "float",
            "half2": "vec2",
            "half3": "vec3",
            "half4": "vec4",
            "packed_half2": "vec2",
            "packed_half3": "vec3",
            "packed_half4": "vec4",
            "float16": "float",
            "f16vec2": "vec2",
            "f16vec3": "vec3",
            "f16vec4": "vec4",
            "min16float": "float",
            "min10float": "float",
            "min16float2": "vec2",
            "min16float3": "vec3",
            "min16float4": "vec4",
            "min10float2": "vec2",
            "min10float3": "vec3",
            "min10float4": "vec4",
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            "str": "int",
            "ivec2": "ivec2",
            "ivec3": "ivec3",
            "ivec4": "ivec4",
            "char": "int",
            "signed char": "int",
            "int8": "int",
            "int8_t": "int",
            "uchar": "uint",
            "unsigned char": "uint",
            "uint8": "uint",
            "uint8_t": "uint",
            "short": "int",
            "signed short": "int",
            "ushort": "uint",
            "unsigned short": "uint",
            "int16": "int",
            "int16_t": "int",
            "int32_t": "int",
            "int64": "int",
            "int64_t": "int",
            "long": "int",
            "signed long": "int",
            "ptrdiff_t": "int",
            "uint16": "uint",
            "uint16_t": "uint",
            "uint32_t": "uint",
            "uint64": "uint",
            "uint64_t": "uint",
            "ulong": "uint",
            "unsigned long": "uint",
            "size_t": "uint",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "packed_int2": "ivec2",
            "packed_int3": "ivec3",
            "packed_int4": "ivec4",
            "simd_int2": "ivec2",
            "simd_int3": "ivec3",
            "simd_int4": "ivec4",
            "char2": "ivec2",
            "char3": "ivec3",
            "char4": "ivec4",
            "i8vec2": "ivec2",
            "i8vec3": "ivec3",
            "i8vec4": "ivec4",
            "i16vec2": "ivec2",
            "i16vec3": "ivec3",
            "i16vec4": "ivec4",
            "uchar2": "uvec2",
            "uchar3": "uvec3",
            "uchar4": "uvec4",
            "u8vec2": "uvec2",
            "u8vec3": "uvec3",
            "u8vec4": "uvec4",
            "u16vec2": "uvec2",
            "u16vec3": "uvec3",
            "u16vec4": "uvec4",
            "short2": "ivec2",
            "short3": "ivec3",
            "short4": "ivec4",
            "ushort2": "uvec2",
            "ushort3": "uvec3",
            "ushort4": "uvec4",
            "min16int": "int",
            "min12int": "int",
            "min16int2": "ivec2",
            "min16int3": "ivec3",
            "min16int4": "ivec4",
            "min12int2": "ivec2",
            "min12int3": "ivec3",
            "min12int4": "ivec4",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "packed_uint2": "uvec2",
            "packed_uint3": "uvec3",
            "packed_uint4": "uvec4",
            "simd_uint2": "uvec2",
            "simd_uint3": "uvec3",
            "simd_uint4": "uvec4",
            "min16uint": "uint",
            "min16uint2": "uvec2",
            "min16uint3": "uvec3",
            "min16uint4": "uvec4",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
            "float2x2": "mat2",
            "float2x3": "mat3x2",
            "float2x4": "mat4x2",
            "float3x2": "mat2x3",
            "float3x3": "mat3",
            "float3x4": "mat4x3",
            "float4x2": "mat2x4",
            "float4x3": "mat3x4",
            "float4x4": "mat4",
            "simd_float2x2": "mat2",
            "simd_float2x3": "mat3x2",
            "simd_float2x4": "mat4x2",
            "simd_float3x2": "mat2x3",
            "simd_float3x3": "mat3",
            "simd_float3x4": "mat4x3",
            "simd_float4x2": "mat2x4",
            "simd_float4x3": "mat3x4",
            "simd_float4x4": "mat4",
            "half2x2": "mat2",
            "half2x3": "mat3x2",
            "half2x4": "mat4x2",
            "half3x2": "mat2x3",
            "half3x3": "mat3",
            "half3x4": "mat4x3",
            "half4x2": "mat2x4",
            "half4x3": "mat3x4",
            "half4x4": "mat4",
            "f16mat2": "mat2",
            "f16mat3": "mat3",
            "f16mat4": "mat4",
            "f16mat2x2": "mat2",
            "f16mat2x3": "mat2x3",
            "f16mat2x4": "mat2x4",
            "f16mat3x2": "mat3x2",
            "f16mat3x3": "mat3",
            "f16mat3x4": "mat3x4",
            "f16mat4x2": "mat4x2",
            "f16mat4x3": "mat4x3",
            "f16mat4x4": "mat4",
            "min16float2x2": "mat2",
            "min16float2x3": "mat3x2",
            "min16float2x4": "mat4x2",
            "min16float3x2": "mat2x3",
            "min16float3x3": "mat3",
            "min16float3x4": "mat4x3",
            "min16float4x2": "mat2x4",
            "min16float4x3": "mat3x4",
            "min16float4x4": "mat4",
            "min10float2x2": "mat2",
            "min10float2x3": "mat3x2",
            "min10float2x4": "mat4x2",
            "min10float3x2": "mat2x3",
            "min10float3x3": "mat3",
            "min10float3x4": "mat4x3",
            "min10float4x2": "mat2x4",
            "min10float4x3": "mat3x4",
            "min10float4x4": "mat4",
            "double2x2": "dmat2",
            "double2x3": "dmat3x2",
            "double2x4": "dmat4x2",
            "double3x2": "dmat2x3",
            "double3x3": "dmat3",
            "double3x4": "dmat4x3",
            "double4x2": "dmat2x4",
            "double4x3": "dmat3x4",
            "double4x4": "dmat4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "double": "double",
            "void": "void",
            "sampler": "sampler",
            "sampler1D": "sampler1D",
            "sampler1DArray": "sampler1DArray",
            "sampler2D": "sampler2D",
            "sampler3D": "sampler3D",
            "samplerCube": "samplerCube",
            "sampler2DArray": "sampler2DArray",
            "samplerCubeArray": "samplerCubeArray",
            "sampler2DMS": "sampler2DMS",
            "sampler2DMSArray": "sampler2DMSArray",
            "sampler2DShadow": "sampler2DShadow",
            "sampler2DArrayShadow": "sampler2DArrayShadow",
            "samplerCubeShadow": "samplerCubeShadow",
            "samplerCubeArrayShadow": "samplerCubeArrayShadow",
            "iimage1D": "iimage1D",
            "iimage1DArray": "iimage1DArray",
            "iimage2D": "iimage2D",
            "iimage3D": "iimage3D",
            "iimage2DArray": "iimage2DArray",
            "iimage2DMS": "iimage2DMS",
            "iimage2DMSArray": "iimage2DMSArray",
            "uimage1D": "uimage1D",
            "uimage1DArray": "uimage1DArray",
            "uimage2D": "uimage2D",
            "uimage3D": "uimage3D",
            "uimage2DArray": "uimage2DArray",
            "uimage2DMS": "uimage2DMS",
            "uimage2DMSArray": "uimage2DMSArray",
            "image1D": "image1D",
            "image1DArray": "image1DArray",
            "image2D": "image2D",
            "image3D": "image3D",
            "imageCube": "imageCube",
            "image2DArray": "image2DArray",
            "image2DMS": "image2DMS",
            "image2DMSArray": "image2DMSArray",
            "accelerationStructureEXT": "accelerationStructureEXT",
        }

        self.function_map = {
            "atan2": "atan",
            "lerp": "mix",
            "frac": "fract",
            "saturate": "clamp",
            "tex2D": "texture",
            "tex2Dproj": "textureProj",
            "tex2Dlod": "textureLod",
            "tex2Dbias": "texture",
            "tex2Dgrad": "textureGrad",
            "tex2Doffset": "textureOffset",
            "texCUBE": "texture",
            "texCUBElod": "textureLod",
            "texCUBEbias": "texture",
            "texCUBEgrad": "textureGrad",
            "textureOffset": "textureOffset",
            "textureProj": "textureProj",
            "textureGatherOffset": "textureGatherOffset",
            "textureGatherOffsets": "textureGatherOffsets",
            "textureQueryLevels": "textureQueryLevels",
            "textureQueryLod": "textureQueryLod",
            "texelFetch": "texelFetch",
            "TraceRay": "traceRayEXT",
            "ReportHit": "reportIntersectionEXT",
            "ReportIntersection": "reportIntersectionEXT",
            "CallShader": "executeCallableEXT",
            "ExecuteCallable": "executeCallableEXT",
            "imageAtomicAdd": "imageAtomicAdd",
            "imageAtomicMin": "imageAtomicMin",
            "imageAtomicMax": "imageAtomicMax",
            "imageAtomicAnd": "imageAtomicAnd",
            "imageAtomicOr": "imageAtomicOr",
            "imageAtomicXor": "imageAtomicXor",
            "imageAtomicExchange": "imageAtomicExchange",
            "imageAtomicCompSwap": "imageAtomicCompSwap",
            "atomicCounterIncrement": "atomicCounterIncrement",
            "atomicCounterDecrement": "atomicCounterDecrement",
            "atomicCounter": "atomicCounter",
            "atomicCounterAdd": "atomicCounterAdd",
            "mul": "*",  # Matrix multiplication
            "ddx": "dFdx",
            "ddy": "dFdy",
            "rsqrt": "inversesqrt",
            "sincos": "sin_cos",  # Custom function needed
            "clip": "discard",  # HLSL clip becomes GLSL discard
            "log2": "log2",
            "exp2": "exp2",
            "pow": "pow",
            "sqrt": "sqrt",
            "abs": "abs",
            "sign": "sign",
            "floor": "floor",
            "ceil": "ceil",
            "round": "round",
            "fmod": "mod",
            "trunc": "trunc",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "step": "step",
            "smoothstep": "smoothstep",
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            "all": "all",
            "any": "any",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
        }

    def generate(self, ast):
        """Generate complete GLSL source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_stage(self, ast, shader_type):
        """Generate GLSL source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def glsl_stage_names(self, ast, target_stage=None):
        stage_names = set()

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if should_emit_qualified_function(target_stage, stage_name):
                stage_names.add(stage_name)

        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_matches(target_stage, stage_name):
                stage_names.add(stage_name)

        return stage_names

    def default_glsl_version_line(self, ast, target_stage=None):
        if self.glsl_stage_names(
            ast, target_stage
        ) & self.RAY_STAGE_NAMES or self.uses_ray_extension_type(ast, target_stage):
            return "#version 460 core"
        return "#version 450 core"

    def glsl_stage_extension_lines(self, ast, target_stage=None):
        stage_names = self.glsl_stage_names(ast, target_stage)

        lines = []
        if stage_names & self.MESH_STAGE_NAMES:
            lines.append("#extension GL_EXT_mesh_shader : require")
        uses_ray_stage = bool(stage_names & self.RAY_STAGE_NAMES)
        if uses_ray_stage:
            lines.append("#extension GL_EXT_ray_tracing : require")
        if self.uses_ray_query(ast, target_stage) or (
            self.uses_acceleration_structure(ast, target_stage) and not uses_ray_stage
        ):
            lines.append("#extension GL_EXT_ray_query : require")
        if self.uses_ray_tracing_position_fetch(ast, target_stage):
            lines.append("#extension GL_EXT_ray_tracing_position_fetch : require")
        return lines

    def uses_ray_extension_type(self, ast, target_stage=None):
        return (
            self.uses_ray_query(ast, target_stage)
            or self.uses_acceleration_structure(ast, target_stage)
            or self.uses_ray_tracing_position_fetch(ast, target_stage)
        )

    def uses_ray_query(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if isinstance(node, RayQueryOpNode):
                    return True
                if any(
                    self.is_ray_query_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    return True
        return False

    def uses_ray_tracing_position_fetch(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if isinstance(node, RayQueryOpNode) and (
                    node.operation in self.RAY_QUERY_POSITION_FETCH_METHODS
                ):
                    return True
                if isinstance(node, FunctionCallNode):
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    if (
                        isinstance(func_expr, MemberAccessNode)
                        and func_expr.member in self.RAY_QUERY_POSITION_FETCH_METHODS
                    ):
                        return True
                    if (
                        self.function_call_name(node)
                        == "rayQueryGetIntersectionTriangleVertexPositionsEXT"
                    ):
                        return True
                if getattr(node, "name", None) == "gl_HitTriangleVertexPositionsEXT":
                    return True
        return False

    def uses_acceleration_structure(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if any(
                    self.is_acceleration_structure_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    return True
        return False

    def ray_extension_type_nodes(self, node):
        return (
            getattr(node, "var_type", None),
            getattr(node, "vtype", None),
            getattr(node, "param_type", None),
            getattr(node, "return_type", None),
        )

    def ray_query_search_roots(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        if target_stage is None:
            return [ast]

        roots = []
        roots.extend(getattr(ast, "global_variables", []) or [])
        roots.extend(getattr(ast, "constants", []) or [])

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if should_emit_qualified_function(target_stage, stage_name):
                roots.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_matches(target_stage, stage_name):
                roots.append(stage)

        return roots

    def generate_program(self, ast, target_stage=None):
        """Render an AST to GLSL, optionally filtering stage entry points."""
        target_stage = normalize_stage_name(target_stage)
        self.validate_stage_main_helper_conflict(ast, target_stage)

        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.current_resource_aliases = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.function_sampler_parameter_indices = (
            self.collect_function_sampler_parameter_indices(ast)
        )
        functions = self.collect_functions(ast)
        self.function_return_types = {
            func.name: self.type_name_string(getattr(func, "return_type", "void"))
            for func in functions
            if getattr(func, "name", None)
        }
        self.function_parameter_names = collect_function_parameter_names(functions)
        self.function_parameter_types = self.collect_function_parameter_types(functions)
        self.function_parameter_infos = self.collect_function_parameter_infos(functions)
        self.function_definitions = {
            func.name: func for func in functions if getattr(func, "name", None)
        }
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        self.emitted_glsl_resource_specialization_names = set()
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                functions,
                self.function_parameter_names,
                self.walk_ast,
                self.function_call_name,
                self.expression_name,
            )
        )
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.current_stage_output_member_map = {}
        self.current_stage_parameter_aliases = {}
        self.current_target_stage = target_stage
        self.stage_io_used_locations = {}
        self.stage_io_declarations = {}
        self.flattened_stage_variables = set()
        self.fragment_output_member_layout_maps = {}
        self.current_function_return_type = None
        self.current_stage_return_type = None
        self.current_stage_entry_type = None
        self.current_expression_expected_type = None
        self.match_temp_variable_index = 0
        self.local_variable_types = {}
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.structured_buffer_instance_members = {}
        self.structured_buffer_counter_members = {}
        self.structured_buffer_counter_instances = {}
        self.glsl_buffer_block_struct_names = set()
        structs = deduplicate_named_declarations(
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            "struct",
        )
        self.struct_member_types = collect_struct_member_types(
            structs, self.type_name_string
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
            functions,
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        if generic_function_specializations:
            self.function_definitions.update(
                {
                    func.name: func
                    for func in generic_function_specializations.values()
                    if getattr(func, "name", None)
                }
            )
            self.function_parameter_names.update(
                collect_function_parameter_names(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_types.update(
                self.collect_function_parameter_types(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_infos.update(
                self.collect_function_parameter_infos(
                    generic_function_specializations.values()
                )
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
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.unsupported_structured_buffer_array_functions = (
            self.collect_unsupported_structured_buffer_array_functions(functions)
        )
        self.validate_global_resource_shadows(ast)
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        version_line = None
        extra_lines = []
        for directive in preprocessors:
            if isinstance(directive, PreprocessorNode):
                if directive.directive == "precision":
                    line = (
                        f"precision {directive.content};"
                        if directive.content
                        else "precision;"
                    )
                else:
                    line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line.startswith("#version") and version_line is None:
                version_line = line
            elif line:
                extra_lines.append(line)
        if version_line is None:
            version_line = self.default_glsl_version_line(ast, target_stage)
        code += f"{version_line}\n"
        for line in self.glsl_stage_extension_lines(ast, target_stage):
            if line not in extra_lines:
                code += f"{line}\n"
        if extra_lines:
            code += "\n".join(extra_lines) + "\n"
        code += generate_enum_constants(
            self,
            self.plain_enums + self.struct_payload_enums,
            qualifier="const",
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
            qualifier="const",
        )
        code += self.generate_constants(ast)
        code += generate_generic_structs(self, self.generic_struct_specializations)
        code += generate_enum_structs(self, self.struct_payload_enums)
        code += generate_generic_enum_structs(self, self.generic_enum_specializations)
        code += generate_enum_constructor_functions(self, self.struct_payload_enums)
        code += generate_generic_enum_constructor_functions(
            self,
            self.generic_enum_specializations,
        )

        global_vars = list(getattr(ast, "global_variables", []) or [])
        stage_local_resource_vars = collect_stage_local_variables(
            ast, target_stage, self.is_stage_local_resource_variable
        )
        stage_local_interface_vars = self.deduplicate_stage_interface_declarations(
            collect_stage_local_variables(
                ast, target_stage, self.is_stage_local_interface_variable
            )
        )
        self.structs_by_name = {
            node.name: node for node in structs if isinstance(node, StructNode)
        }
        stage_resource_params = self.collect_stage_entry_resource_parameters(
            ast, target_stage
        )
        resource_declaration_nodes = self.deduplicate_resource_declaration_nodes(
            global_vars + stage_local_resource_vars + stage_resource_params
        )
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(resource_declaration_nodes)
        )
        self.vertex_input_struct_names = self.stage_parameter_struct_names(
            ast, "vertex"
        )
        self.vertex_output_struct_names = self.stage_return_struct_names(ast, "vertex")
        self.fragment_input_struct_names = self.stage_parameter_struct_names(
            ast, "fragment"
        )
        self.fragment_output_struct_names = self.stage_return_struct_names(
            ast, "fragment"
        )
        self.vertex_input_member_names = self.struct_member_names(
            self.vertex_input_struct_names
        )
        self.vertex_input_member_names.update(
            self.stage_value_parameter_input_names(ast, "vertex")
        )
        self.combined_vertex_output_member_names = (
            self.vertex_output_declaration_names()
        )
        self.fragment_input_member_names = self.fragment_input_declaration_names(
            self.fragment_input_struct_names
        )
        self.fragment_input_member_names.update(
            self.stage_value_parameter_input_names(ast, "fragment")
        )
        self.fragment_output_member_name_maps = self.fragment_output_member_maps()
        self.fragment_output_member_layout_maps = (
            self.fragment_output_member_layout_maps_for_outputs()
        )
        emit_vertex_io = target_stage in {None, "vertex"}
        emit_fragment_io = target_stage in {None, "fragment"}
        emit_graphics_io = target_stage in {None, "vertex", "fragment"}
        for node in structs:
            if isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                if node.name in self.generic_struct_definitions:
                    continue
                if node.name in self.glsl_buffer_block_struct_names:
                    continue
                elif (
                    node.name == "VSInput"
                    or node.name in self.vertex_input_struct_names
                ):
                    if emit_vertex_io:
                        code += self.generate_stage_input_declarations(node)
                    elif not emit_graphics_io:
                        code += self.generate_struct(node)
                elif node.name == "VSOutput":
                    emitted_io = False
                    if node.name in self.vertex_output_struct_names and emit_vertex_io:
                        code += self.generate_vertex_output_declarations(node)
                        emitted_io = True
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        code += self.generate_fragment_input_declarations(node)
                        emitted_io = True
                    if (
                        node.name in self.fragment_output_struct_names
                        and emit_fragment_io
                    ):
                        code += self.generate_fragment_output_declarations(node)
                        emitted_io = True
                    if not emitted_io:
                        if emit_vertex_io:
                            code += self.generate_legacy_output_declarations(node)
                        else:
                            code += self.generate_struct(node)
                    elif node.name in self.fragment_output_struct_names:
                        code += self.generate_struct(node)
                elif node.name in self.vertex_output_struct_names:
                    if emit_vertex_io:
                        code += self.generate_vertex_output_declarations(node)
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        code += self.generate_fragment_input_declarations(node)
                    if (
                        node.name in self.fragment_output_struct_names
                        and emit_fragment_io
                    ):
                        code += self.generate_fragment_output_declarations(node)
                    code += self.generate_struct(node)
                elif node.name == "PSInput":
                    if emit_fragment_io:
                        code += self.generate_fragment_input_declarations(node)
                    elif not emit_graphics_io:
                        code += self.generate_struct(node)
                elif node.name in self.fragment_input_struct_names:
                    if emit_fragment_io:
                        code += self.generate_fragment_input_declarations(node)
                        if node.name in self.fragment_output_struct_names:
                            code += self.generate_fragment_output_declarations(node)
                    if node.name in self.fragment_output_struct_names:
                        code += self.generate_struct(node)
                    elif not emit_graphics_io:
                        code += self.generate_struct(node)
                elif node.name in self.fragment_output_struct_names:
                    if emit_fragment_io:
                        code += self.generate_fragment_output_declarations(node)
                    if node.name != "PSOutput" or not emit_fragment_io:
                        code += self.generate_struct(node)
                elif node.name == "PSOutput":
                    if emit_fragment_io:
                        code += self.generate_fragment_output_declarations(node)
                    else:
                        code += self.generate_struct(node)
                else:
                    code += self.generate_struct(node)

        code += self.generate_stage_parameter_input_declarations(ast, target_stage)

        resource_binding_cursors = {}
        used_resource_bindings = {}
        self.reserve_explicit_global_resource_bindings(
            resource_declaration_nodes, used_resource_bindings
        )
        for index, node in enumerate(resource_declaration_nodes):
            vtype, array_size, array_suffix, resource_count = (
                self.resource_declaration_shape(node)
            )

            if hasattr(node, "name"):
                var_name = node.name
            elif hasattr(node, "variable_name"):
                var_name = node.variable_name
            else:
                var_name = f"var{index}"

            if self.is_glsl_buffer_block_variable(node, vtype):
                if self.is_shader_record_buffer_block(node):
                    if self.explicit_resource_binding_index(node) is not None:
                        raise ValueError(
                            "GLSL shaderRecordEXT buffer blocks cannot declare "
                            "binding layout qualifiers"
                        )
                    declaration = self.glsl_buffer_block_declaration(
                        node, vtype, var_name, None, array_suffix
                    )
                    if declaration is not None:
                        code += declaration
                        continue

                binding_namespace = "buffer binding"
                explicit_binding = self.explicit_resource_binding_index(node)
                resource_binding = (
                    explicit_binding
                    if explicit_binding is not None
                    else self.next_available_resource_binding(
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                )
                declaration = self.glsl_buffer_block_declaration(
                    node, vtype, var_name, resource_binding, array_suffix
                )
                if declaration is not None:
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        resource_binding,
                        resource_count,
                        var_name,
                    )
                    code += declaration
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        resource_binding,
                        resource_count,
                    )
                    continue

            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type == "sampler":
                self.sampler_variables.add(var_name)
                continue
            if self.is_structured_buffer_type(vtype):
                binding_namespace = "buffer binding"
                explicit_binding = self.explicit_resource_binding_index(node)
                resource_binding = (
                    explicit_binding
                    if explicit_binding is not None
                    else self.next_available_resource_binding(
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                code += self.structured_buffer_block_declaration(
                    vtype, var_name, resource_binding, array_size
                )
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                if self.structured_buffer_requires_counter(vtype):
                    counter_binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                    counter_name = self.structured_buffer_counter_resource_name(
                        var_name
                    )
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        counter_binding,
                        resource_count,
                        counter_name,
                    )
                    code += self.structured_buffer_counter_block_declaration(
                        var_name, counter_binding, array_size
                    )
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        counter_binding,
                        resource_count,
                    )
                continue
            if self.is_opaque_resource_type(mapped_type):
                self.texture_variable_types[var_name] = mapped_type
                record_explicit_image_metadata(
                    var_name,
                    node,
                    self.attribute_value_to_string,
                    image_formats=self.image_variable_formats,
                    image_accesses=self.image_variable_accesses,
                )
            declaration = format_c_style_array_declaration(
                f"{mapped_type}{array_suffix}", var_name
            )
            if self.is_opaque_resource_type(mapped_type):
                binding_namespace = (
                    "image binding"
                    if self.is_storage_image_type(vtype)
                    else "texture binding"
                )
                explicit_binding = self.explicit_resource_binding_index(node)
                resource_binding = (
                    explicit_binding
                    if explicit_binding is not None
                    else self.next_available_resource_binding(
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                layout = self.opaque_resource_layout(
                    mapped_type, resource_binding, node
                )
                memory_qualifiers = self.resource_memory_qualifiers(node)
                qualifier_prefix = f"{memory_qualifiers} " if memory_qualifiers else ""
                code += f"{layout} {qualifier_prefix}uniform {declaration};\n"
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
            else:
                code += self.generate_global_variable_declaration(
                    node, declaration, vtype
                )

        for node in stage_local_interface_vars:
            code += self.generate_stage_local_interface_variable_declaration(node)

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        combined_stage_entry_names = self.combined_stage_entry_names(ast, target_stage)
        self.prepare_glsl_resource_function_specializations(ast)

        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)
            qualifier_name = normalize_stage_name(qualifier)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            resource_specializations = self.glsl_resource_function_emission_list(
                getattr(func, "name", None)
            )
            if resource_specializations and qualifier_name not in {
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
            }:
                for specialized_func in resource_specializations:
                    code += self.generate_function(specialized_func)

            if generic_function_parameters(func):
                for specialized_func in generic_function_emission_list(self, func):
                    code += self.generate_function(specialized_func)
                continue

            if qualifier_name == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(
                    func,
                    shader_type="compute",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                stage_code = ""
                for func in order_functions_by_dependencies(
                    getattr(stage, "local_functions", []) or [],
                    self.walk_ast,
                    self.function_call_name,
                    FunctionCallNode,
                ):
                    resource_specializations = (
                        self.glsl_resource_function_emission_list(
                            getattr(func, "name", None)
                        )
                    )
                    if resource_specializations:
                        for specialized_func in resource_specializations:
                            stage_code += self.generate_function(specialized_func)

                    if generic_function_parameters(func):
                        for specialized_func in generic_function_emission_list(
                            self,
                            func,
                        ):
                            stage_code += self.generate_function(specialized_func)
                    else:
                        stage_code += self.generate_function(func)

                if hasattr(stage, "entry_point"):
                    stage_code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                        stage_layout_qualifiers=getattr(
                            stage, "layout_qualifiers", None
                        ),
                        entry_name=combined_stage_entry_names.get(
                            id(stage.entry_point)
                        ),
                    )

                if stage_code:
                    code += f"// {stage_name.title()} Shader\n"
                    code += stage_code

        return code

    def prepare_glsl_resource_function_specializations(self, ast):
        """Create GLSL helper variants for concrete storage image arguments."""
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        pending = []

        def scan_function(func, aliases):
            before = set(self.glsl_resource_function_specializations)
            self.collect_glsl_resource_specializations_from_body(
                getattr(func, "body", []),
                aliases,
            )
            after = set(self.glsl_resource_function_specializations)
            pending.extend(after - before)

        for func in self.collect_functions(ast):
            scan_function(func, {})

        seen = set()
        while pending:
            key = pending.pop(0)
            if key in seen:
                continue
            seen.add(key)
            specialized_func = self.glsl_resource_function_specializations.get(key)
            if specialized_func is None:
                continue
            scan_function(
                specialized_func,
                getattr(specialized_func, "_glsl_resource_aliases", {}) or {},
            )

        if not self.glsl_resource_function_specializations:
            return

        specialized_functions = list(
            self.glsl_resource_function_specializations.values()
        )
        self.function_definitions.update(
            {
                func.name: func
                for func in specialized_functions
                if getattr(func, "name", None)
            }
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in specialized_functions
                if getattr(func, "name", None)
            }
        )
        self.function_parameter_names.update(
            collect_function_parameter_names(specialized_functions)
        )
        self.function_parameter_types.update(
            self.collect_function_parameter_types(specialized_functions)
        )
        self.function_parameter_infos.update(
            self.collect_function_parameter_infos(specialized_functions)
        )
        self.function_sampler_parameter_indices.update(
            self.collect_function_sampler_parameter_indices(specialized_functions)
        )

    def collect_glsl_resource_specializations_from_body(self, body, aliases):
        for node in self.walk_ast(body):
            if not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_name(node)
            if not func_name:
                continue
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            self.ensure_glsl_resource_function_specialization(
                func_name,
                args,
                aliases,
            )

    def glsl_resource_function_emission_list(self, source_name):
        if not source_name:
            return []
        functions = [
            func
            for func in self.glsl_resource_function_specializations.values()
            if getattr(func, "_glsl_resource_source_name", None) == source_name
        ]
        functions.sort(key=lambda func: getattr(func, "name", ""))
        result = []
        for func in functions:
            name = getattr(func, "name", None)
            if name in self.emitted_glsl_resource_specialization_names:
                continue
            self.emitted_glsl_resource_specialization_names.add(name)
            result.append(func)
        return result

    def glsl_resource_binding_info(self, arg, aliases):
        arg_name = self.expression_name(arg)
        if arg_name in aliases:
            return aliases[arg_name]

        resource_type = self.texture_argument_resource_type(arg)
        if not self.is_storage_image_type(resource_type):
            return None

        expression = self.glsl_resource_argument_expression(arg, aliases)
        if expression is None:
            return None

        return {
            "expression": expression,
            "type": resource_type,
            "format": self.image_resource_format(arg),
            "access": self.image_resource_access(arg),
        }

    def glsl_resource_argument_expression(self, arg, aliases):
        arg_name = self.expression_name(arg)
        if arg_name in aliases:
            return aliases[arg_name]["expression"]
        if isinstance(arg, str):
            return arg
        if hasattr(arg, "name") and isinstance(arg.name, str):
            return arg.name
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_code = self.glsl_resource_argument_expression(array_expr, aliases)
            if array_code is None:
                return None
            return f"{array_code}[{self.generate_expression(index_expr)}]"
        return None

    def glsl_resource_function_specialization_key(self, func_name, args, aliases):
        callee = self.function_definitions.get(func_name)
        if callee is None:
            return None, None

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        access_requirements = self.function_image_access_requirements.get(func_name, {})
        bindings = {}
        key_parts = []
        for index, (param, arg) in enumerate(zip(params, args or [])):
            param_type = self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            if not self.is_storage_image_type(param_type):
                continue
            binding = self.glsl_resource_binding_info(arg, aliases)
            if binding is None:
                continue
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            required_access = access_requirements.get(param_name)
            if required_access is not None and not image_access_satisfies_requirement(
                required_access,
                binding.get("access"),
            ):
                return None, None
            bindings[index] = (param_name, binding)
            key_parts.append((index, binding["expression"]))

        if not bindings:
            return None, None
        return (func_name, tuple(key_parts)), bindings

    def ensure_glsl_resource_function_specialization(self, func_name, args, aliases):
        key, bindings = self.glsl_resource_function_specialization_key(
            func_name,
            args,
            aliases,
        )
        if key is None:
            return None
        if key in self.glsl_resource_function_specializations:
            return self.glsl_resource_function_specializations[key]

        source_func = self.function_definitions.get(func_name)
        if source_func is None:
            return None

        clone = deepcopy(source_func)
        clone.name = self.glsl_resource_specialization_name(func_name, bindings)
        clone.parameters = [
            param
            for index, param in enumerate(
                getattr(clone, "parameters", getattr(clone, "params", []))
            )
            if index not in bindings
        ]
        clone._glsl_resource_source_name = func_name
        clone._glsl_resource_bound_indices = set(bindings)
        clone._glsl_resource_aliases = {
            param_name: binding for param_name, binding in bindings.values()
        }

        self.glsl_resource_function_specializations[key] = clone
        self.glsl_resource_function_call_names[key] = clone.name
        self.glsl_resource_specialized_source_names.add(func_name)
        return clone

    def glsl_resource_specialization_name(self, func_name, bindings):
        suffix_parts = []
        for _, (param_name, binding) in sorted(bindings.items()):
            suffix_parts.append(
                "{}_{}".format(
                    sanitize_type_name(param_name),
                    sanitize_type_name(binding["expression"]),
                )
            )
        suffix = "_".join(suffix_parts)
        return "{}__glsl_{}".format(sanitize_type_name(func_name), suffix)

    def glsl_resource_function_call_specialization(self, func_name, args):
        key, _ = self.glsl_resource_function_specialization_key(
            func_name,
            args,
            self.current_resource_aliases,
        )
        if key is None:
            return None
        return self.glsl_resource_function_specializations.get(key)

    def glsl_resource_specialized_call_arguments(self, specialized_func, args):
        bound_indices = getattr(specialized_func, "_glsl_resource_bound_indices", set())
        return [
            arg for index, arg in enumerate(args or []) if index not in bound_indices
        ]

    def generate_constants(self, ast):
        code = ""
        for node in getattr(ast, "constants", []) or []:
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = self.generate_constant_expression(value)
            code += f"const {self.map_type(const_type)} {name} = {value_code};\n"

        return f"{code}\n" if code else ""

    def generate_constant_expression(self, expr):
        value_code = self.generate_expression(expr)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def combined_stage_entry_types(self):
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
        }

    def combined_stage_entry_names(self, ast, target_stage=None):
        """Return distinct helper names for monolithic multi-stage output."""
        if target_stage is not None:
            return {}

        stage_entry_types = self.combined_stage_entry_types()
        entries = collect_stage_entry_records(ast, None, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, None, stage_entry_types
        )
        return assign_stage_entry_names(
            entries,
            used_names,
            lambda stage_name, _func: f"{stage_name}_main",
            single_entry_default="main",
        )

    def validate_stage_main_helper_conflict(self, ast, target_stage):
        if target_stage is None:
            return

        stage_entry_types = self.combined_stage_entry_types()
        has_global_main_helper = False
        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if hasattr(func, "qualifiers") and func.qualifiers
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if stage_name in stage_entry_types:
                continue
            if getattr(func, "name", None) == "main":
                has_global_main_helper = True
                break

        if not has_global_main_helper:
            return

        if collect_stage_entry_records(ast, target_stage, stage_entry_types):
            raise ValueError(
                "Global helper function 'main' conflicts with the GLSL "
                f"{target_stage} stage entry point"
            )

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate cbuffer name(s) in OpenGL output: {names}")

        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(ast)
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise ValueError(
                f"Cbuffer name(s) conflict with existing OpenGL declaration(s): {names}"
            )

        duplicate_members = collect_duplicate_cbuffer_member_names(cbuffers)
        if duplicate_members:
            names = ", ".join(sorted(duplicate_members))
            raise ValueError(
                f"Ambiguous cbuffer member name(s) in OpenGL output: {names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(ast)
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise ValueError(
                "Cbuffer member name(s) conflict with OpenGL global declaration(s): "
                f"{names}"
            )
        cbuffer_binding_cursors = {}
        used_cbuffer_bindings = {}
        for node in cbuffers:
            explicit_binding = self.explicit_resource_binding_index(node)
            if explicit_binding is None:
                continue
            self.reserve_resource_binding_range(
                used_cbuffer_bindings,
                "OpenGL",
                "uniform buffer binding",
                explicit_binding,
                1,
                node.name,
            )
        for node in cbuffers:
            explicit_binding = self.explicit_resource_binding_index(node)
            resource_binding = (
                explicit_binding
                if explicit_binding is not None
                else self.next_available_resource_binding(
                    used_cbuffer_bindings,
                    cbuffer_binding_cursors,
                    "uniform buffer binding",
                    1,
                )
            )
            self.reserve_resource_binding_range(
                used_cbuffer_bindings,
                "OpenGL",
                "uniform buffer binding",
                resource_binding,
                1,
                node.name,
            )
            self.advance_resource_binding(
                cbuffer_binding_cursors,
                "uniform buffer binding",
                resource_binding,
                1,
            )
            if isinstance(node, StructNode):
                code += f"layout(std140, binding = {resource_binding}) uniform {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[];\n"
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
                code += f"layout(std140, binding = {resource_binding}) uniform {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[];\n"
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

    def generate_compute_layout(self, execution_config=None):
        x, y, z = compute_local_size(execution_config)
        return (
            f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\n"
        )

    def generate_stage_layout(self, shader_type, func, stage_layout_qualifiers=None):
        if shader_type == "fragment":
            return self.generate_fragment_stage_layout(func, stage_layout_qualifiers)
        if shader_type == "geometry":
            return self.generate_geometry_stage_layout(func, stage_layout_qualifiers)
        if shader_type == "tessellation_control":
            return self.generate_tessellation_control_layout(
                func, stage_layout_qualifiers
            )
        if shader_type == "tessellation_evaluation":
            return self.generate_tessellation_evaluation_layout(
                func, stage_layout_qualifiers
            )
        if shader_type == "mesh":
            return self.generate_mesh_stage_layout(func, stage_layout_qualifiers)
        return ""

    def generate_fragment_stage_layout(self, func, stage_layout_qualifiers=None):
        if self.glsl_stage_bare_attribute(
            func, {"early_fragment_tests"}, stage_layout_qualifiers, "in"
        ):
            return "layout(early_fragment_tests) in;\n"
        return ""

    def generate_geometry_stage_layout(self, func, stage_layout_qualifiers=None):
        input_primitive = self.glsl_stage_bare_attribute(
            func,
            {
                "points",
                "lines",
                "lines_adjacency",
                "triangles",
                "triangles_adjacency",
            },
            stage_layout_qualifiers,
            "in",
        )
        output_primitive = self.glsl_single_stage_attribute_argument(
            func, "outputtopology", stage_layout_qualifiers, "out"
        )
        if output_primitive is None:
            output_primitive = self.glsl_stage_bare_attribute(
                func,
                {"points", "line_strip", "triangle_strip"},
                stage_layout_qualifiers,
                "out",
            )
        max_vertices = self.glsl_single_stage_attribute_argument(
            func, "max_vertices", stage_layout_qualifiers, "out"
        )
        if max_vertices is None:
            max_vertices = self.glsl_single_stage_attribute_argument(
                func, "maxvertexcount", stage_layout_qualifiers, "out"
            )
        invocations = self.glsl_single_stage_attribute_argument(
            func, "invocations", stage_layout_qualifiers, "in"
        )

        code = ""
        input_layout = []
        if input_primitive:
            input_layout.append(input_primitive)
        if invocations is not None:
            input_layout.append(f"invocations = {invocations}")
        if input_layout:
            code += f"layout({', '.join(input_layout)}) in;\n"

        output_layout = []
        if output_primitive:
            output_layout.append(self.glsl_geometry_output_topology(output_primitive))
        if max_vertices is not None:
            output_layout.append(f"max_vertices = {max_vertices}")
        if output_layout:
            code += f"layout({', '.join(output_layout)}) out;\n"
        return code

    def generate_tessellation_control_layout(self, func, stage_layout_qualifiers=None):
        vertices = self.glsl_single_stage_attribute_argument(
            func, "vertices", stage_layout_qualifiers, "out"
        )
        if vertices is None:
            vertices = self.glsl_single_stage_attribute_argument(
                func, "outputcontrolpoints", stage_layout_qualifiers, "out"
            )
        if vertices is None:
            return ""
        return f"layout(vertices = {vertices}) out;\n"

    def generate_tessellation_evaluation_layout(
        self, func, stage_layout_qualifiers=None
    ):
        layout_parts = []
        domain = self.glsl_single_stage_attribute_argument(
            func, "domain", stage_layout_qualifiers, "in"
        )
        if domain is None:
            domain = self.glsl_stage_bare_attribute(
                func, {"triangles", "quads", "isolines"}, stage_layout_qualifiers, "in"
            )
        if domain:
            layout_parts.append(self.glsl_tessellation_domain(domain))

        partitioning = self.glsl_single_stage_attribute_argument(
            func, "partitioning", stage_layout_qualifiers, "in"
        )
        if partitioning is None:
            partitioning = self.glsl_stage_bare_attribute(
                func,
                {
                    "equal_spacing",
                    "fractional_even_spacing",
                    "fractional_odd_spacing",
                },
                stage_layout_qualifiers,
                "in",
            )
        if partitioning:
            layout_parts.append(self.glsl_tessellation_partitioning(partitioning))

        winding = self.glsl_stage_bare_attribute(
            func, {"cw", "ccw"}, stage_layout_qualifiers, "in"
        )
        if winding:
            layout_parts.append(winding)

        if self.glsl_stage_bare_attribute(
            func, {"point_mode"}, stage_layout_qualifiers, "in"
        ):
            layout_parts.append("point_mode")

        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) in;\n"

    def generate_mesh_stage_layout(self, func, stage_layout_qualifiers=None):
        output_primitive = self.glsl_stage_bare_attribute(
            func, {"points", "lines", "triangles"}, stage_layout_qualifiers, "out"
        )
        if output_primitive is None:
            output_primitive = self.glsl_single_stage_attribute_argument(
                func, "outputtopology", stage_layout_qualifiers, "out"
            )

        max_vertices = self.glsl_single_stage_attribute_argument(
            func, "max_vertices", stage_layout_qualifiers, "out"
        )
        if max_vertices is None:
            max_vertices = self.glsl_single_stage_attribute_argument(
                func, "maxvertexcount", stage_layout_qualifiers, "out"
            )

        max_primitives = self.glsl_single_stage_attribute_argument(
            func, "max_primitives", stage_layout_qualifiers, "out"
        )
        if max_primitives is None:
            max_primitives = self.glsl_single_stage_attribute_argument(
                func, "maxprimitivecount", stage_layout_qualifiers, "out"
            )

        layout_parts = []
        if output_primitive:
            layout_parts.append(self.glsl_mesh_output_topology(output_primitive))
        if max_vertices is not None:
            layout_parts.append(f"max_vertices = {max_vertices}")
        if max_primitives is not None:
            layout_parts.append(f"max_primitives = {max_primitives}")

        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) out;\n"

    def glsl_stage_bare_attribute(
        self, func, names, stage_layout_qualifiers=None, direction=None
    ):
        names = set(names)
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names and not getattr(attr, "arguments", []):
                return attr_name
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, direction):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names and not getattr(attr, "arguments", []):
                return attr_name
        return None

    def glsl_single_stage_attribute_argument(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        arguments = self.glsl_stage_attribute_arguments(
            func, attribute_name, stage_layout_qualifiers, direction
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"GLSL stage attribute {attribute_name} requires exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def glsl_stage_attribute_arguments(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        for attr in getattr(func, "attributes", []) or []:
            if self.glsl_stage_control_attribute_name(attr) == attribute_name:
                return getattr(attr, "arguments", []) or []
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, direction):
            if self.glsl_stage_control_attribute_name(attr) == attribute_name:
                return getattr(attr, "arguments", []) or []
        return []

    def glsl_stage_layout_entries(self, stage_layout_qualifiers, direction=None):
        if direction is not None:
            direction = str(direction).lower()
        for layout in stage_layout_qualifiers or []:
            layout_direction = getattr(layout, "direction", None)
            if layout_direction is not None:
                layout_direction = str(layout_direction).lower()
            if direction is not None and layout_direction != direction:
                continue
            yield from getattr(layout, "entries", []) or []

    def glsl_geometry_output_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "line_strip",
            "lines": "line_strip",
            "line_strip": "line_strip",
            "triangle": "triangle_strip",
            "triangles": "triangle_strip",
            "triangle_strip": "triangle_strip",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise ValueError(
                f"GLSL geometry output topology cannot be lowered: {topology}"
            )
        return mapped

    def glsl_mesh_output_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "lines",
            "lines": "lines",
            "triangle": "triangles",
            "triangles": "triangles",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise ValueError(f"GLSL mesh output topology cannot be lowered: {topology}")
        return mapped

    def map_mesh_intrinsic(self, operation):
        mesh_intrinsics = {
            "SetMeshOutputCounts": "SetMeshOutputsEXT",
            "DispatchMesh": "EmitMeshTasksEXT",
        }
        return mesh_intrinsics.get(operation, operation)

    def map_ray_tracing_intrinsic(self, operation, args):
        ray_intrinsics = {
            "TraceRay": "traceRayEXT",
            "ReportHit": "reportIntersectionEXT",
            "CallShader": "executeCallableEXT",
            "AcceptHitAndEndSearch": "terminateRayEXT",
            "IgnoreHit": "ignoreIntersectionEXT",
        }
        mapped = ray_intrinsics.get(operation, operation)
        if mapped in {"ignoreIntersectionEXT", "terminateRayEXT"} and not args:
            return mapped
        return f"{mapped}({', '.join(args)})"

    def map_ray_query_intrinsic(self, operation, query, args):
        mapping = self.RAY_QUERY_METHOD_MAP.get(operation)
        if mapping is None:
            return f"{query}.{operation}({', '.join(args)})"

        function_name, committed = mapping
        call_args = [query]
        if committed is not None:
            call_args.append(committed)
        call_args.extend(args)
        return f"{function_name}({', '.join(call_args)})"

    def glsl_tessellation_domain(self, domain):
        domain_name = str(domain).strip().strip('"').lower()
        domain_map = {
            "tri": "triangles",
            "triangle": "triangles",
            "triangles": "triangles",
            "quad": "quads",
            "quads": "quads",
            "isoline": "isolines",
            "isolines": "isolines",
        }
        mapped = domain_map.get(domain_name)
        if mapped is None:
            raise ValueError(f"GLSL tessellation domain cannot be lowered: {domain}")
        return mapped

    def glsl_tessellation_partitioning(self, partitioning):
        partition_name = str(partitioning).strip().strip('"').lower()
        partition_map = {
            "equal": "equal_spacing",
            "equal_spacing": "equal_spacing",
            "fractional_even": "fractional_even_spacing",
            "fractional_even_spacing": "fractional_even_spacing",
            "fractional_odd": "fractional_odd_spacing",
            "fractional_odd_spacing": "fractional_odd_spacing",
        }
        mapped = partition_map.get(partition_name)
        if mapped is None:
            raise ValueError(
                f"GLSL tessellation partitioning cannot be lowered: {partitioning}"
            )
        return mapped

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        stage_layout_qualifiers=None,
    ):
        """Render a function or GLSL ``main`` stage entry point."""
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        image_format_parameters = {}
        image_access_parameters = {}
        resource_aliases = getattr(func, "_glsl_resource_aliases", {}) or {}
        unsupported_buffer_array_info = (
            self.unsupported_structured_buffer_array_functions.get(func.name, {})
        )
        unsupported_buffer_array_indices = unsupported_buffer_array_info.get(
            "indices", set()
        )
        previous_function_return_type = self.current_function_return_type
        previous_local_variable_types = self.local_variable_types
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        previous_structured_buffer_array_parameters = (
            self.current_structured_buffer_array_parameters
        )
        previous_structured_buffer_counter_parameters = (
            self.current_structured_buffer_counter_parameters
        )
        self.local_variable_types = {}
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        for alias_name, binding in resource_aliases.items():
            self.local_variable_types[alias_name] = binding.get("type")
        for index, p in enumerate(param_list):
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

            if index in unsupported_buffer_array_indices:
                continue

            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
                continue

            buffer_array = self.structured_buffer_array_parameter_info(
                raw_param_type, p.name, getattr(func, "name", None)
            )
            if buffer_array is not None:
                expanded_names = [
                    f"{p.name}_{index}" for index in range(buffer_array["count"])
                ]
                buffer_info = {
                    **buffer_array,
                    "expanded_names": expanded_names,
                }
                if self.structured_buffer_requires_counter(buffer_array["base_type"]):
                    counter_names = [
                        f"{p.name}Counter_{index}"
                        for index in range(buffer_array["count"])
                    ]
                    buffer_info["counter_expanded_names"] = counter_names
                self.current_structured_buffer_array_parameters[p.name] = buffer_info
                for expanded_name in expanded_names:
                    self.local_variable_types[expanded_name] = (
                        f"{buffer_array['element_type']}[]"
                    )
                    params.append(
                        format_c_style_array_declaration(
                            f"{buffer_array['element_type']}[]",
                            expanded_name,
                        )
                    )
                for counter_name in buffer_info.get("counter_expanded_names", []):
                    params.append(
                        format_c_style_array_declaration("uint[]", counter_name)
                    )
                continue

            param_type = self.map_resource_parameter_type_with_hint(
                raw_param_type, p, getattr(func, "name", None)
            )
            if self.is_opaque_resource_type(
                self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
            ):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=image_format_parameters,
                    image_accesses=image_access_parameters,
                )

            semantic = self.semantic_from_node(p)

            declaration = format_c_style_array_declaration(param_type, p.name)
            if self.is_storage_image_type(param_type):
                memory_qualifiers = self.resource_memory_qualifiers(p)
                if memory_qualifiers:
                    declaration = f"{memory_qualifiers} {declaration}"
            semantic_attr = self.map_semantic(semantic)
            params.append(
                f"{declaration} {semantic_attr}" if semantic_attr else declaration
            )
            if self.structured_buffer_requires_counter(raw_param_type):
                counter_name = self.structured_buffer_counter_parameter_name(p.name)
                self.current_structured_buffer_counter_parameters[p.name] = counter_name
                params.append(format_c_style_array_declaration("uint[]", counter_name))

        if shader_type == "compute":
            self.validate_compute_builtin_parameter_types(param_list)
        elif shader_type is not None:
            self.validate_graphics_builtin_parameter_types(param_list, shader_type)

        params_str = ", ".join(params)

        stage_entry_types = {
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
        }

        if shader_type is not None:
            self.validate_function_return_semantic(func, shader_type)

        stage_output = self.stage_return_output(func, shader_type)
        if stage_output and stage_output["declaration"]:
            code += f"{stage_output['declaration']}\n"

        stage_layout = self.generate_stage_layout(
            shader_type, func, stage_layout_qualifiers
        )
        if stage_layout:
            code += stage_layout

        if shader_type in {"compute", "mesh", "task"}:
            code += self.generate_compute_layout(execution_config)

        if shader_type in stage_entry_types:
            code += f"void {entry_name or 'main'}() {{\n"
            self.current_function_return_type = "void"
        else:
            raw_return_type = self.type_name_string(getattr(func, "return_type", None))
            self.current_function_return_type = raw_return_type or "void"
            return_type = self.map_type(self.current_function_return_type)
            code += f"{return_type} {func.name}({params_str}) {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_resource_aliases = self.current_resource_aliases
        previous_identifier_aliases = self.current_identifier_aliases
        previous_image_format_parameters = self.current_image_format_parameters
        previous_image_access_parameters = self.current_image_access_parameters
        previous_stage_output = self.current_stage_output
        previous_stage_inputs = self.current_stage_inputs
        previous_stage_outputs = self.current_stage_outputs
        previous_stage_output_member_map = self.current_stage_output_member_map
        previous_stage_parameter_aliases = self.current_stage_parameter_aliases
        previous_flattened_stage_variables = self.flattened_stage_variables
        previous_stage_return_type = self.current_stage_return_type
        previous_stage_entry_type = self.current_stage_entry_type
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = {
            **texture_parameters,
            **{
                alias_name: binding.get("type")
                for alias_name, binding in resource_aliases.items()
            },
        }
        self.current_resource_aliases = resource_aliases
        self.current_identifier_aliases = {}
        self.current_image_format_parameters = {
            **image_format_parameters,
            **{
                alias_name: binding.get("format")
                for alias_name, binding in resource_aliases.items()
                if binding.get("format") is not None
            },
        }
        self.current_image_access_parameters = {
            **image_access_parameters,
            **{
                alias_name: binding.get("access")
                for alias_name, binding in resource_aliases.items()
                if binding.get("access") is not None
            },
        }
        self.current_stage_output = stage_output
        self.current_stage_inputs = self.stage_input_member_maps(func, shader_type)
        self.current_stage_output_member_map = self.stage_output_member_map(
            func, shader_type
        )
        self.current_stage_outputs = self.stage_output_member_maps(func, shader_type)
        self.current_stage_parameter_aliases = self.stage_parameter_aliases(
            func, shader_type
        )
        self.flattened_stage_variables = set(self.current_stage_outputs)
        self.current_stage_return_type = self.type_node_name(
            getattr(func, "return_type", None)
        )
        self.current_stage_entry_type = shader_type
        body = getattr(func, "body", [])
        if unsupported_buffer_array_info:
            code += self.unsupported_structured_buffer_array_function_body(
                func, unsupported_buffer_array_info, 1
            )
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, 1)
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_texture_parameters = previous_texture_parameters
        self.current_resource_aliases = previous_resource_aliases
        self.current_identifier_aliases = previous_identifier_aliases
        self.current_image_format_parameters = previous_image_format_parameters
        self.current_image_access_parameters = previous_image_access_parameters
        self.current_stage_output = previous_stage_output
        self.current_stage_inputs = previous_stage_inputs
        self.current_stage_outputs = previous_stage_outputs
        self.current_stage_output_member_map = previous_stage_output_member_map
        self.current_stage_parameter_aliases = previous_stage_parameter_aliases
        self.flattened_stage_variables = previous_flattened_stage_variables
        self.current_stage_return_type = previous_stage_return_type
        self.current_stage_entry_type = previous_stage_entry_type
        self.current_function_return_type = previous_function_return_type
        self.local_variable_types = previous_local_variable_types
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.current_structured_buffer_array_parameters = (
            previous_structured_buffer_array_parameters
        )
        self.current_structured_buffer_counter_parameters = (
            previous_structured_buffer_counter_parameters
        )

        code += "}\n\n"
        return code

    def stage_functions(self, ast, stage_name):
        functions = []

        for func in getattr(ast, "functions", []) or []:
            qualifiers = getattr(func, "qualifiers", []) or []
            qualifier = (
                qualifiers[0] if qualifiers else getattr(func, "qualifier", None)
            )
            if normalize_stage_name(qualifier) == stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            current_stage = normalize_stage_name(stage_type)
            if current_stage == stage_name and hasattr(stage, "entry_point"):
                functions.append(stage.entry_point)

        return functions

    def stage_parameter_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
            for param in parameters:
                type_name = self.type_node_name(getattr(param, "param_type", None))
                if type_name in self.structs_by_name:
                    struct_names.add(type_name)
        return struct_names

    def stage_return_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            type_name = self.type_node_name(getattr(func, "return_type", None))
            if type_name in self.structs_by_name:
                struct_names.add(type_name)
        return struct_names

    def collect_stage_entry_resource_parameters(self, ast, target_stage=None):
        stage_entry_types = {
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
        }
        parameters = []
        seen = set()

        def add_parameters(func):
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                if id(param) in seen:
                    continue
                if not self.is_stage_entry_resource_parameter(param):
                    continue
                parameters.append(param)
                seen.add(id(param))

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if hasattr(func, "qualifiers") and func.qualifiers
                else getattr(func, "qualifier", None)
            )
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name not in stage_entry_types:
                continue
            if should_emit_qualified_function(target_stage, qualifier_name):
                add_parameters(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                add_parameters(entry_point)

        return parameters

    def is_stage_local_resource_variable(self, node):
        vtype = self.type_name_string(
            getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        base_type = self.resource_base_type(vtype)
        mapped_type = self.map_resource_type_with_format(base_type, node)
        return (
            mapped_type == "sampler"
            or self.is_opaque_resource_type(mapped_type)
            or self.is_structured_buffer_type(vtype)
            or self.is_glsl_buffer_block_variable(node, vtype)
        )

    def is_stage_local_interface_variable(self, node):
        if self.is_stage_local_resource_variable(node):
            return False
        return bool(self.glsl_variable_qualifier_prefix(node))

    def deduplicate_stage_interface_declarations(self, nodes):
        declarations = []
        declarations_by_name = {}
        for node in nodes:
            name = self.resource_node_name(node)
            if not name:
                declarations.append(node)
                continue

            vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
            signature = (
                self.map_type(vtype),
                array_suffix,
                self.glsl_variable_layout_prefix(node),
                self.glsl_variable_qualifier_prefix(node),
            )
            existing = declarations_by_name.get(name)
            if existing is None:
                declarations_by_name[name] = signature
                declarations.append(node)
                continue
            if existing != signature:
                raise ValueError(
                    "Conflicting OpenGL stage interface declaration for "
                    f"'{name}': {signature} differs from {existing}"
                )
        return declarations

    def deduplicate_resource_declaration_nodes(self, nodes):
        declarations = []
        declarations_by_name = {}
        for node in nodes:
            name = self.resource_node_name(node)
            identity = self.resource_declaration_identity(node)
            if not name or identity is None:
                declarations.append(node)
                continue

            binding = self.explicit_resource_binding_index(node)
            existing = declarations_by_name.get(name)
            if existing is None:
                declarations_by_name[name] = {
                    "identity": identity,
                    "binding": binding,
                    "index": len(declarations),
                }
                declarations.append(node)
                continue

            if existing["identity"] != identity:
                raise ValueError(
                    "Conflicting OpenGL resource declaration for "
                    f"'{name}': {identity} differs from {existing['identity']}"
                )

            existing_binding = existing["binding"]
            if (
                existing_binding is not None
                and binding is not None
                and existing_binding != binding
            ):
                raise ValueError(
                    "Conflicting OpenGL resource binding for "
                    f"'{name}': binding {binding} differs from "
                    f"existing binding {existing_binding}"
                )
            if existing_binding is None and binding is not None:
                declarations[existing["index"]] = node
                existing["binding"] = binding

        return declarations

    def is_stage_entry_resource_parameter(self, param):
        raw_type = self.resource_node_type(param)
        if self.is_sampler_type(raw_type):
            return False
        if self.is_glsl_buffer_block_variable(param, raw_type):
            return True
        if self.is_structured_buffer_type(raw_type):
            return True
        mapped_type = self.map_resource_type_with_format(
            self.resource_base_type(raw_type), param
        )
        return self.is_opaque_resource_type(mapped_type)

    def is_stage_entry_value_parameter(self, param):
        raw_type = self.resource_node_type(param)
        if self.is_sampler_type(raw_type):
            return False
        if self.is_stage_entry_resource_parameter(param):
            return False
        type_name = self.type_node_name(raw_type)
        return type_name not in self.structs_by_name

    def is_stage_builtin_semantic(self, semantic):
        mapped_semantic = self.map_semantic(semantic)
        return mapped_semantic.startswith("gl_")

    def parameter_has_mapped_semantic(self, parameter, expected_semantic):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return False
        return self.map_semantic(semantic).lower() == expected_semantic.lower()

    def validate_exact_mapped_semantic_type(
        self, parameters, stage_name, semantic, expected_type, expected_description
    ):
        for parameter in parameters:
            if not self.parameter_has_mapped_semantic(parameter, semantic):
                continue

            mapped_type = self.map_type(self.resource_node_type(parameter))
            base_type, array_suffix = split_array_type_suffix(str(mapped_type))
            if array_suffix or base_type != expected_type:
                raise ValueError(
                    f"OpenGL {stage_name} stage {semantic} "
                    f"parameter '{parameter.name}' "
                    f"must be {expected_description}"
                )

    def validate_compute_builtin_parameter_types(self, parameters):
        for semantic in (
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        ):
            self.validate_exact_mapped_semantic_type(
                parameters, "compute", semantic, "uvec3", "uvec3"
            )
        self.validate_exact_mapped_semantic_type(
            parameters,
            "compute",
            "gl_LocalInvocationIndex",
            "uint",
            "scalar uint",
        )

    def validate_graphics_builtin_parameter_types(self, parameters, stage_name):
        for semantic in ("gl_VertexID", "gl_InstanceID", "gl_PrimitiveID"):
            self.validate_exact_mapped_semantic_type(
                parameters, stage_name, semantic, "int", "scalar int"
            )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_FrontFacing", "bool", "scalar bool"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_FragCoord", "vec4", "vec4"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_PointCoord", "vec2", "vec2"
        )

    def validate_function_return_semantic(self, func, stage_name):
        semantic = self.function_return_semantic(func)
        if semantic is None:
            return
        mapped_semantic = self.map_semantic(semantic)
        return_type = self.function_return_type(func)
        return_base_type, array_suffix = split_array_type_suffix(str(return_type))
        if not array_suffix and return_base_type == "void":
            function_name = getattr(func, "name", "<anonymous>")
            raise ValueError(
                f"OpenGL {stage_name} function '{function_name}' cannot use "
                f"return semantic '{semantic}' with void return type"
            )

        self.validate_function_return_semantic_stage(stage_name, semantic)

        invalid_return_semantics = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_FrontFacing",
            "gl_FragCoord",
            "gl_PointCoord",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        if mapped_semantic in invalid_return_semantics:
            raise ValueError(
                f"OpenGL {stage_name} semantic '{semantic}' cannot be used as "
                "a function return semantic"
            )

    def validate_function_return_semantic_stage(self, stage_name, semantic):
        mapped_semantic = self.map_semantic(semantic)
        semantic_text = str(semantic)
        is_fragment_output = (
            semantic_text == "gl_FragDepth" or semantic_text.startswith("gl_FragColor")
        )
        is_vertex_output = self.is_vertex_builtin_output(mapped_semantic)

        if stage_name == "vertex" and is_fragment_output:
            raise ValueError(
                f"OpenGL vertex stage function return cannot use fragment output "
                f"semantic '{semantic}'"
            )
        if stage_name == "fragment" and is_vertex_output:
            raise ValueError(
                f"OpenGL fragment stage function return cannot use vertex output "
                f"semantic '{semantic}'"
            )
        if stage_name == "compute" and (is_fragment_output or is_vertex_output):
            raise ValueError(
                f"OpenGL compute stage function return cannot use graphics output "
                f"semantic '{semantic}'"
            )

    def generate_stage_parameter_input_declarations(self, ast, target_stage=None):
        declarations = []
        graphics_stages = {
            "vertex",
            "fragment",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
        }

        def add_parameter(param, stage_name):
            if not self.is_stage_entry_value_parameter(param):
                return
            semantic = self.semantic_from_node(param)
            if self.is_stage_builtin_semantic(semantic):
                return

            param_type = self.map_type(self.resource_node_type(param))
            layout = self.map_semantic(semantic)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                stage_name,
                "input",
                layout,
                param_type,
                param.name,
            )
            prefix = f"{layout} " if layout.startswith("layout(") else ""
            parameter_declaration = format_c_style_array_declaration(
                param_type, param.name
            )
            declaration = f"{prefix}in {parameter_declaration};"
            if self.reserve_stage_io_declaration(
                stage_name, "input", param.name, declaration
            ):
                declarations.append(declaration)

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if hasattr(func, "qualifiers") and func.qualifiers
                else getattr(func, "qualifier", None)
            )
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name not in graphics_stages:
                continue
            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                add_parameter(param, qualifier_name)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in graphics_stages:
                continue
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            for param in (
                getattr(entry_point, "parameters", getattr(entry_point, "params", []))
                or []
            ):
                add_parameter(param, stage_name)

        return "\n".join(declarations) + ("\n" if declarations else "")

    def stage_io_layout_location(self, layout):
        if not layout.startswith("layout("):
            return None
        start = layout.find("(")
        end = layout.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return None

        for item in layout[start + 1 : end].split(","):
            key, separator, value = item.partition("=")
            if separator and key.strip() == "location":
                value = value.strip()
                return int(value) if value.isdigit() else None
        return None

    def stage_io_location_count(self, mapped_type):
        base_type, array_size = parse_array_type(str(mapped_type))
        array_count = max(array_size or 1, 1)
        matrix_count = 1

        matrix_size = ""
        if base_type.startswith("dmat"):
            matrix_size = base_type[4:]
        elif base_type.startswith("mat"):
            matrix_size = base_type[3:]

        if matrix_size:
            columns = matrix_size.split("x", 1)[0]
            if columns.isdigit():
                matrix_count = int(columns)

        return max(matrix_count * array_count, 1)

    def reserve_stage_io_layout(
        self, used_locations, stage_name, direction, layout, mapped_type, name
    ):
        start = self.stage_io_layout_location(layout)
        if start is None:
            return

        count = self.stage_io_location_count(mapped_type)
        end = start + count - 1
        ranges = used_locations.setdefault((stage_name, direction), [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting OpenGL {stage_name} {direction} location "
                    f"for '{name}': {self.stage_io_location_label(start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.stage_io_location_label(used_start, used_end)}"
                )
        ranges.append((start, end, name))

    def reserve_stage_io_declaration(self, stage_name, direction, name, declaration):
        key = (stage_name, direction, name)
        existing = self.stage_io_declarations.get(key)
        if existing is None:
            self.stage_io_declarations[key] = declaration
            return True
        if existing == declaration:
            return False
        raise ValueError(
            f"Conflicting OpenGL {stage_name} {direction} declaration for "
            f"'{name}': {declaration} differs from {existing}"
        )

    def stage_io_declared_names(self, stage_name, direction):
        return {
            name
            for stage, used_direction, name in self.stage_io_declarations
            if stage == stage_name and used_direction == direction
        }

    def stage_io_location_label(self, start, end):
        if start == end:
            return f"location {start}"
        return f"locations {start}-{end}"

    def struct_member_names(self, struct_names):
        names = set()
        for struct_name in struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                names.add(member.name)
        return names

    def fragment_input_declaration_names(self, struct_names):
        names = set()
        for struct_name in struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                input_name = self.fragment_input_member_name(member, struct_name)
                if input_name is not None:
                    names.add(input_name)
        return names

    def vertex_output_declaration_names(self):
        names = set()
        for struct_name in self.vertex_output_struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                output_name = self.vertex_output_member_name(member)
                if not self.is_vertex_builtin_output(output_name):
                    names.add(output_name)
        return names

    def stage_value_parameter_input_names(self, ast, stage_name):
        names = set()
        for func in self.stage_functions(ast, stage_name):
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                if not self.is_stage_entry_value_parameter(param):
                    continue
                if self.is_stage_builtin_semantic(self.semantic_from_node(param)):
                    continue
                names.add(param.name)
        return names

    def stage_io_name_avoiding_reserved(self, preferred_name, reserved_names):
        if preferred_name not in reserved_names:
            return preferred_name

        base_name = f"out_{preferred_name}"
        candidate = base_name
        suffix = 1
        while candidate in reserved_names:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        return candidate

    def fragment_output_member_maps(self):
        maps = {}
        reserved_names = set(self.fragment_input_member_names)
        if self.current_target_stage is None:
            reserved_names.update(self.combined_vertex_output_member_names)
        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_map = {}
            for member in getattr(struct, "members", []) or []:
                output_name = self.fragment_output_member_name(
                    member, reserved_names=reserved_names
                )
                member_map[member.name] = output_name
                if output_name != "gl_FragDepth":
                    reserved_names.add(output_name)
            maps[struct_name] = member_map
        return maps

    def next_available_stage_io_layout(self, stage_name, direction, mapped_type):
        count = self.stage_io_location_count(mapped_type)
        ranges = sorted(self.stage_io_used_locations.get((stage_name, direction), []))
        location = 0

        while True:
            end = location + count - 1
            overlap = next(
                (
                    (used_start, used_end)
                    for used_start, used_end, _ in ranges
                    if location <= used_end and used_start <= end
                ),
                None,
            )
            if overlap is None:
                return f"layout(location = {location})"
            location = overlap[1] + 1

    def fragment_output_member_layout_maps_for_outputs(self):
        maps = {}

        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_layouts = {}
            for member in getattr(struct, "members", []) or []:
                output_name = self.fragment_output_member_name(member, struct_name)
                mapped_semantic = self.map_semantic(self.semantic_from_node(member))
                if mapped_semantic == "gl_FragDepth" or output_name == "gl_FragDepth":
                    member_layouts[member.name] = "gl_FragDepth"
                    continue
                if not mapped_semantic.startswith("layout("):
                    continue

                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    "fragment",
                    "output",
                    mapped_semantic,
                    self.member_type_name(member),
                    output_name,
                )
                member_layouts[member.name] = mapped_semantic

            maps[struct_name] = member_layouts

        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_layouts = maps.setdefault(struct_name, {})
            for member in getattr(struct, "members", []) or []:
                if member.name in member_layouts:
                    continue

                output_name = self.fragment_output_member_name(member, struct_name)
                layout = self.next_available_stage_io_layout(
                    "fragment", "output", self.member_type_name(member)
                )
                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    "fragment",
                    "output",
                    layout,
                    self.member_type_name(member),
                    output_name,
                )
                member_layouts[member.name] = layout

        return maps

    def type_node_name(self, type_node):
        if type_node is None:
            return None
        if hasattr(type_node, "name"):
            return type_node.name
        return str(type_node)

    def generate_stage_input_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_type = self.member_type_name(member)
            semantic = self.semantic_from_node(member)
            layout = self.map_semantic(semantic)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "vertex",
                "input",
                layout,
                member_type,
                member.name,
            )
            prefix = f"{layout} " if layout.startswith("layout(") else ""
            declaration = f"{prefix}in {member_type} {member.name};"
            if self.reserve_stage_io_declaration(
                "vertex", "input", member.name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_legacy_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            declaration = f"out {self.member_type_name(member)} {member.name};"
            if self.reserve_stage_io_declaration(
                "vertex", "output", member.name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_vertex_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_builtin_output(output_name):
                continue

            semantic = self.semantic_from_node(member)
            layout = self.map_semantic(semantic)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "vertex",
                "output",
                layout,
                self.member_type_name(member),
                output_name,
            )
            prefix = f"{layout} " if layout.startswith("layout(") else ""
            declaration = f"{prefix}out {self.member_type_name(member)} {output_name};"
            if self.reserve_stage_io_declaration(
                "vertex", "output", output_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_fragment_input_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            input_name = self.fragment_input_member_name(member, node.name)
            if input_name is None:
                continue

            semantic = self.semantic_from_node(member)
            layout = self.map_semantic(semantic)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "fragment",
                "input",
                layout,
                self.member_type_name(member),
                input_name,
            )
            prefix = f"{layout} " if layout.startswith("layout(") else ""
            declaration = f"{prefix}in {self.member_type_name(member)} {input_name};"
            if self.reserve_stage_io_declaration(
                "fragment", "input", input_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_fragment_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_type = self.member_type_name(member)
            output_name = self.fragment_output_member_name(member, node.name)
            if output_name == "gl_FragDepth":
                continue
            layout = self.fragment_output_member_layout_maps.get(node.name, {}).get(
                member.name
            )
            if layout is None:
                layout = self.map_semantic(self.semantic_from_node(member))
                if not layout.startswith("layout("):
                    layout = "layout(location = 0)"
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "fragment",
                "output",
                layout,
                member_type,
                output_name,
            )
            declaration = f"{layout} out {member_type} {output_name};"
            if self.reserve_stage_io_declaration(
                "fragment", "output", output_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def member_type_name(self, member):
        if hasattr(member, "member_type"):
            return self.map_type(member.member_type)
        return self.map_type(getattr(member, "vtype", "float"))

    def vertex_output_member_name(self, member):
        semantic = self.semantic_from_node(member)
        mapped_semantic = self.map_semantic(semantic)
        if self.is_vertex_builtin_output(mapped_semantic):
            return mapped_semantic
        if member.name in self.vertex_input_member_names:
            return f"out_{member.name}"
        return member.name

    def fragment_output_member_name(
        self, member, struct_name=None, reserved_names=None
    ):
        if struct_name in self.fragment_output_member_name_maps:
            return self.fragment_output_member_name_maps[struct_name].get(
                member.name, member.name
            )

        mapped_semantic = self.map_semantic(self.semantic_from_node(member))
        if mapped_semantic == "gl_FragDepth":
            return mapped_semantic

        if reserved_names is None:
            reserved_names = self.fragment_input_member_names
        return self.stage_io_name_avoiding_reserved(member.name, reserved_names)

    def fragment_input_member_name(self, member, struct_name):
        if struct_name in self.vertex_output_struct_names:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_builtin_output(output_name):
                return None
            return self.combined_fragment_input_member_name(output_name)
        return self.combined_fragment_input_member_name(member.name)

    def combined_fragment_input_member_name(self, input_name):
        if (
            self.current_target_stage is not None
            or input_name not in self.combined_vertex_output_member_names
        ):
            return input_name
        candidate = f"in_{input_name}"
        suffix = 1
        reserved_names = set(self.combined_vertex_output_member_names)
        while candidate in reserved_names:
            suffix += 1
            candidate = f"in_{input_name}_{suffix}"
        return candidate

    def is_vertex_builtin_output(self, name):
        return name in {"gl_Position", "gl_PointSize", "gl_ClipDistance"}

    def stage_input_member_maps(self, func, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return {}

        maps = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            type_name = self.type_node_name(getattr(param, "param_type", None))
            struct = self.structs_by_name.get(type_name)
            if struct is None:
                continue
            if shader_type == "fragment":
                member_map = {}
                for member in getattr(struct, "members", []) or []:
                    input_name = self.fragment_input_member_name(member, type_name)
                    if input_name is not None:
                        member_map[member.name] = input_name
                maps[param.name] = member_map
            else:
                maps[param.name] = {
                    member.name: member.name
                    for member in getattr(struct, "members", [])
                }
        return maps

    def stage_parameter_aliases(self, func, shader_type):
        if shader_type is None:
            return {}

        aliases = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            semantic = self.semantic_from_node(param)
            if self.is_stage_builtin_semantic(semantic):
                aliases[param.name] = self.map_semantic(semantic)
        return aliases

    def stage_output_member_map(self, func, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return {}

        type_name = self.type_node_name(getattr(func, "return_type", None))
        struct = self.structs_by_name.get(type_name)
        if struct is None:
            return {}

        if shader_type == "fragment":
            return {
                member.name: self.fragment_output_member_name(member, type_name)
                for member in getattr(struct, "members", []) or []
            }
        return {
            member.name: self.vertex_output_member_name(member)
            for member in getattr(struct, "members", []) or []
        }

    def stage_output_member_maps(self, func, shader_type):
        member_map = self.stage_output_member_map(func, shader_type)
        if not member_map:
            return {}

        type_name = self.type_node_name(getattr(func, "return_type", None))
        maps = {}
        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        for stmt in statements:
            if not isinstance(stmt, VariableNode):
                continue
            if self.type_node_name(getattr(stmt, "var_type", None)) == type_name:
                maps[stmt.name] = member_map
        return maps

    def function_return_type(self, func):
        return_type = getattr(func, "return_type", None)
        if return_type is None:
            return "void"
        return self.map_type(return_type)

    def stage_return_output(self, func, shader_type):
        if shader_type == "fragment":
            return self.fragment_stage_output(func, shader_type)
        if shader_type == "vertex":
            return self.vertex_stage_output(func)
        return None

    def is_void_stage_entry_return_value(self):
        if self.current_stage_output is not None:
            return False
        if self.current_stage_entry_type is None:
            return False
        return self.current_function_return_type == "void"

    def vertex_stage_output(self, func):
        output_type = self.function_return_type(func)
        if output_type == "void":
            return None

        semantic = self.function_return_semantic(func)
        mapped_semantic = self.map_semantic(semantic)
        if not mapped_semantic and output_type == "vec4":
            mapped_semantic = "gl_Position"
        elif not mapped_semantic:
            if output_type in self.structs_by_name or output_type not in {
                "float",
                "double",
                "int",
                "uint",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "dvec2",
                "dvec3",
                "dvec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "mat2",
                "mat3",
                "mat4",
                "dmat2",
                "dmat3",
                "dmat4",
            }:
                return None
            raise ValueError(
                "OpenGL vertex stage entry point with non-void return type "
                f"'{output_type}' requires an output semantic"
            )

        if self.is_vertex_builtin_output(mapped_semantic):
            return {
                "name": mapped_semantic,
                "declaration": "",
            }

        layout = mapped_semantic
        if not layout.startswith("layout("):
            layout = "layout(location = 0)"
        reserved_names = set(self.vertex_input_member_names)
        reserved_names.update(self.stage_io_declared_names("vertex", "output"))
        output_name = self.stage_io_name_avoiding_reserved(
            "vertexOutput", reserved_names
        )
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            "vertex",
            "output",
            layout,
            output_type,
            output_name,
        )
        declaration = f"{layout} out {output_type} {output_name};"
        if not self.reserve_stage_io_declaration(
            "vertex", "output", output_name, declaration
        ):
            declaration = ""
        return {
            "name": output_name,
            "declaration": declaration,
        }

    def fragment_stage_output(self, func, shader_type):
        if shader_type != "fragment":
            return None

        output_type = self.function_return_type(func)
        if output_type == "void":
            return None
        if output_type in self.structs_by_name:
            return None

        semantic = self.function_return_semantic(func) or "gl_FragColor"
        if semantic == "gl_FragDepth":
            return {
                "name": "gl_FragDepth",
                "declaration": "",
            }

        layout = self.map_semantic(semantic)
        if not layout.startswith("layout("):
            layout = "layout(location = 0)"

        output_name = self.fragment_output_name(semantic)
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            "fragment",
            "output",
            layout,
            output_type,
            output_name,
        )
        declaration = f"{layout} out {output_type} {output_name};"
        if not self.reserve_stage_io_declaration(
            "fragment", "output", output_name, declaration
        ):
            declaration = ""
        return {
            "name": output_name,
            "declaration": declaration,
        }

    def fragment_output_name(self, semantic):
        if semantic and semantic.startswith("gl_FragColor"):
            suffix = semantic[len("gl_FragColor") :]
            output_name = f"fragColor{suffix}"
        else:
            output_name = "fragColor"

        reserved_names = set(self.fragment_input_member_names)
        for member_map in self.fragment_output_member_name_maps.values():
            for name in member_map.values():
                if name != "gl_FragDepth":
                    reserved_names.add(name)
        reserved_names.update(self.stage_io_declared_names("fragment", "output"))
        return self.stage_io_name_avoiding_reserved(output_name, reserved_names)

    def generate_stage_struct_constructor_return(self, expr, indent):
        if not self.current_stage_output_member_map:
            return None

        struct = self.structs_by_name.get(self.current_stage_return_type)
        if struct is None:
            return None

        members = getattr(struct, "members", []) or []
        field_values = self.stage_struct_constructor_field_values(expr, members)
        if field_values is None:
            return None

        indent_str = "    " * indent
        code = ""
        for member, value_expr in field_values:
            output_name = self.current_stage_output_member_map.get(member.name)
            if output_name is None:
                continue
            value = self.generate_expression_with_expected(
                value_expr, self.member_type_name(member)
            )
            code += f"{indent_str}{output_name} = {value};\n"
        code += f"{indent_str}return;\n"
        return code

    def stage_struct_constructor_field_values(self, expr, members):
        if isinstance(expr, FunctionCallNode):
            if self.function_call_name(expr) != self.current_stage_return_type:
                return None
            args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            if len(args) != len(members):
                return None
            return list(zip(members, args))

        if isinstance(expr, ConstructorNode):
            constructor_type = self.type_name_string(
                getattr(expr, "constructor_type", None)
            )
            if constructor_type != self.current_stage_return_type:
                return None
            args = list(getattr(expr, "arguments", []) or [])
            named_args = dict(getattr(expr, "named_arguments", {}) or {})
            if len(args) > len(members):
                return None

            values = []
            for index, member in enumerate(members):
                if index < len(args):
                    values.append((member, args[index]))
                    continue
                if member.name not in named_args:
                    return None
                values.append((member, named_args[member.name]))
            return values

        return None

    def local_identifier_name(self, name):
        if name in self.current_identifier_aliases:
            return self.current_identifier_aliases[name]
        if name not in self.GLSL_RESERVED_IDENTIFIERS:
            return name

        used_names = set(self.local_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        used_names.update(self.current_resource_aliases)
        used_names.update(self.current_stage_parameter_aliases.values())

        candidate = f"{name}_"
        while candidate in used_names or candidate in self.GLSL_RESERVED_IDENTIFIERS:
            candidate += "_"
        self.current_identifier_aliases[name] = candidate
        return candidate

    def expression_identifier_name(self, name):
        if name in self.current_identifier_aliases:
            return self.current_identifier_aliases[name]
        return self.current_stage_parameter_aliases.get(name, name)

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as GLSL source."""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            if stmt.name in self.flattened_stage_variables:
                return ""
            var_type = self.local_variable_declared_type(stmt)
            local_name = self.local_identifier_name(stmt.name)
            self.local_variable_types[stmt.name] = var_type
            if local_name != stmt.name:
                self.local_variable_types[local_name] = var_type

            declaration = format_c_style_array_declaration(
                self.map_type(var_type), local_name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            initial_value = getattr(stmt, "initial_value", None)
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    local_name,
                    var_type,
                    indent,
                    "GLSL",
                )
                return code
            if initial_value is not None:
                init_expr = self.generate_expression_with_expected(
                    initial_value, var_type
                )
                return f"{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"
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
        elif isinstance(stmt, ReturnNode):
            if getattr(stmt, "value", None) is None:
                return f"{indent_str}return;\n"
            return_value_name = self.expression_name(stmt.value)
            if return_value_name in self.flattened_stage_variables:
                return f"{indent_str}return;\n"
            stage_struct_return = self.generate_stage_struct_constructor_return(
                stmt.value, indent
            )
            if stage_struct_return is not None:
                return stage_struct_return
            if self.current_stage_output is not None:
                if isinstance(stmt.value, list):
                    values = ", ".join(
                        self.generate_expression(val) for val in stmt.value
                    )
                    value = values
                else:
                    value = self.generate_expression_with_expected(
                        stmt.value, self.current_function_return_type
                    )
                return (
                    f"{indent_str}{self.current_stage_output['name']} = {value};\n"
                    f"{indent_str}return;\n"
                )
            if self.is_void_stage_entry_return_value():
                return f"{indent_str}return;\n"
            if isinstance(stmt.value, list):
                # Multiple return values
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values};\n"
            else:
                return (
                    f"{indent_str}return "
                    f"{self.generate_expression_with_expected(stmt.value, self.current_function_return_type)};\n"
                )
        elif hasattr(stmt, "__class__") and "ExpressionStatementNode" in str(
            type(stmt)
        ):
            # Handle ExpressionStatementNode
            tail_return = self.generate_tail_expression_statement(stmt, indent)
            if tail_return is not None:
                return tail_return
            expr_code = self.generate_expression_statement(stmt)
            return f"{indent_str}{expr_code};\n"
        else:
            # Handle expressions that may be used as statements
            expr_result = self.generate_expression(stmt)
            if expr_result.strip():
                return f"{indent_str}{expr_result};\n"
            else:
                return f"{indent_str}// Unhandled statement: {type(stmt).__name__}\n"

    def generate_tail_expression_statement(self, stmt, indent=0):
        if not getattr(stmt, "is_tail_expression", False):
            return None
        value_expr = getattr(stmt, "expression", None)

        stage_struct_return = self.generate_stage_struct_constructor_return(
            value_expr,
            indent,
        )
        if stage_struct_return is not None:
            return stage_struct_return

        indent_str = "    " * indent
        if self.current_stage_output is not None:
            value = self.generate_expression_with_expected(
                value_expr,
                self.current_function_return_type,
            )
            return (
                f"{indent_str}{self.current_stage_output['name']} = {value};\n"
                f"{indent_str}return;\n"
            )

        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            return None

        value = self.generate_expression_with_expected(value_expr, return_type)
        return f"{indent_str}return {value};\n"

    def local_variable_declared_type(self, stmt):
        var_type = getattr(stmt, "var_type", None)
        if var_type is None:
            var_type = getattr(stmt, "vtype", None)
        if var_type is None:
            var_type = self.expression_result_type(getattr(stmt, "initial_value", None))
        return self.type_name_string(var_type) or "float"

    def local_variable_qualifier(self, node):
        return "const " if "const" in getattr(node, "qualifiers", []) else ""

    def global_variable_qualifier(self, node):
        qualifier_prefix = self.glsl_variable_qualifier_prefix(node)
        if qualifier_prefix:
            return f"{qualifier_prefix} "

        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if qualifiers & {"shared", "groupshared", "workgroup", "threadgroup"}:
            return "shared "
        if "const" in qualifiers:
            return "const "
        return "uniform "

    def generate_global_variable_declaration(self, node, declaration, vtype):
        qualifier = self.global_variable_qualifier(node)
        initializer = ""
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            initializer = (
                f" = {self.generate_expression_with_expected(initial_value, vtype)}"
            )
        layout = self.glsl_variable_layout_prefix(node)
        return f"{layout}{qualifier}{declaration}{initializer};\n"

    def generate_stage_local_interface_variable_declaration(self, node):
        vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
        declaration = format_c_style_array_declaration(
            f"{self.map_type(vtype)}{array_suffix}", self.resource_node_name(node, "")
        )
        return self.generate_global_variable_declaration(node, declaration, vtype)

    def glsl_variable_qualifier_prefix(self, node):
        qualifiers = [
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        ]
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.append(str(attr_name).lower())

        emitted = []

        def add(qualifier):
            if qualifier not in emitted:
                emitted.append(qualifier)

        for qualifier in qualifiers:
            normalized = (
                qualifier[len("glsl_") :]
                if qualifier.startswith("glsl_")
                else qualifier
            )
            normalized = normalized.replace("-", "_")
            if normalized in {"perprimitive", "perprimitiveext"}:
                add("perprimitiveEXT")
            elif normalized in {
                "task_payload_shared",
                "taskpayloadshared",
                "taskpayloadsharedext",
            }:
                add("taskPayloadSharedEXT")
            elif normalized in {"raypayload", "raypayloadext"}:
                add("rayPayloadEXT")
            elif normalized in {"raypayloadin", "raypayloadinext"}:
                add("rayPayloadInEXT")
            elif normalized in {"hitattribute", "hitattributeext"}:
                add("hitAttributeEXT")
            elif normalized in {"callabledata", "callabledataext"}:
                add("callableDataEXT")
            elif normalized in {"callabledatain", "callabledatainext"}:
                add("callableDataInEXT")
            elif normalized in {"shared", "groupshared", "workgroup", "threadgroup"}:
                add("shared")
            elif normalized in {
                "patch",
                "flat",
                "smooth",
                "noperspective",
                "centroid",
                "sample",
                "in",
                "out",
            }:
                add(normalized)

        if not emitted:
            return ""

        order = {
            "patch": 0,
            "perprimitiveEXT": 1,
            "flat": 2,
            "smooth": 3,
            "noperspective": 4,
            "centroid": 5,
            "sample": 6,
            "in": 7,
            "out": 8,
            "shared": 9,
            "taskPayloadSharedEXT": 10,
            "rayPayloadEXT": 11,
            "rayPayloadInEXT": 12,
            "hitAttributeEXT": 13,
            "callableDataEXT": 14,
            "callableDataInEXT": 15,
        }
        emitted.sort(key=lambda qualifier: order.get(qualifier, len(order)))
        return " ".join(emitted)

    def glsl_variable_layout_prefix(self, node):
        layout_parts = []
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower()
            if normalized != "location":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            layout_parts.append(
                f"location = {self.attribute_value_to_string(arguments[0])}"
            )
        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) "

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

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
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("dvec"):
            return "double"
        if mapped_type.startswith("uvec"):
            return "uint"
        if mapped_type.startswith("ivec"):
            return "int"
        if mapped_type.startswith("bvec"):
            return "bool"
        if mapped_type.startswith("vec"):
            return "float"
        return None

    def glsl_constructor_type(self, func_name):
        constructor = self.map_type(func_name) if isinstance(func_name, str) else None
        if constructor in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
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
            "dmat2",
            "dmat3",
            "dmat4",
            "dmat2x2",
            "dmat2x3",
            "dmat2x4",
            "dmat3x2",
            "dmat3x3",
            "dmat3x4",
            "dmat4x2",
            "dmat4x3",
            "dmat4x4",
        }:
            return constructor
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, VariableNode):
            return self.local_variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode) or (
            hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__)
        ):
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
        if isinstance(expr, AssignmentNode):
            target = getattr(expr, "target", getattr(expr, "left", None))
            return self.expression_result_type(target)
        if isinstance(expr, ArrayAccessNode):
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(expr.object)
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
        if isinstance(expr, ConstructorNode):
            return infer_enum_constructor_type(
                self, expr
            ) or infer_struct_constructor_type(self, expr)
        if isinstance(expr, MatchNode):
            return infer_match_expression_result_type(self, expr)
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type:
                return numeric_result_type
            if func_name in {"normalize", "reflect"} and args:
                return self.expression_result_type(args[0])
            if func_name == "dot" and args:
                return (
                    self.vector_component_type(self.expression_result_type(args[0]))
                    or "float"
                )
            if func_name == "cross" and args:
                return self.expression_result_type(args[0])
            specialized_func_name = generic_function_call_name(self, func_name, args)
            if specialized_func_name in self.function_return_types:
                return self.function_return_types[specialized_func_name]
            if func_name in self.function_return_types:
                return self.function_return_types[func_name]
            if func_name == "imageLoad" and args:
                return self.image_load_result_type(args[0])
            constructor = self.glsl_constructor_type(func_name)
            if constructor:
                return constructor
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.local_variable_types.get(getattr(expr, "name", None))
        return None

    def generate_assignment(self, node, is_main=False):
        left_node = getattr(node, "target", getattr(node, "left", None))
        right_node = getattr(node, "value", getattr(node, "right", None))
        left = self.generate_expression(left_node)
        right = self.generate_expression_with_expected(
            right_node, self.expression_result_type(left_node)
        )
        op = self.map_operator(getattr(node, "operator", getattr(node, "op", "=")))
        return f"{left} {op} {right}"

    def generate_if(self, node, indent, is_main=False):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

        if_body = node.if_body
        code += self.generate_scoped_statement_body(if_body, indent + 1)

        code += f"{indent_str}}}"

        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                condition = self.generate_expression(else_if_condition)
                code += f" else if ({condition}) {{\n"
                code += self.generate_scoped_statement_body(else_if_body, indent + 1)
                code += f"{indent_str}}}"

        if hasattr(node, "else_body") and node.else_body:
            code += " else {\n"
            else_body = node.else_body
            code += self.generate_scoped_statement_body(else_body, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent, is_main=False):
        indent_str = "    " * indent
        previous_local_variable_types = dict(self.local_variable_types)
        previous_identifier_aliases = dict(self.current_identifier_aliases)

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

            body = node.body
            code += self.generate_scoped_statement_body(body, indent + 1)

            code += f"{indent_str}}}\n"

            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_identifier_aliases = previous_identifier_aliases

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")
        previous_local_variable_types = dict(self.local_variable_types)
        previous_identifier_aliases = dict(self.current_identifier_aliases)

        try:
            self.local_variable_types[pattern] = "int"

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
            self.current_identifier_aliases = previous_identifier_aliases

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
        if is_switch_lowerable_match(node):
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "GLSL")

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
        previous_identifier_aliases = dict(self.current_identifier_aliases)
        try:
            return self.generate_statement_body(body, indent)
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_identifier_aliases = previous_identifier_aliases

    def generate_statement_body(self, body, indent):
        code = ""
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent)
        elif body is not None:
            code += self.generate_statement(body, indent)
        return code

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, str):
            return init
        if isinstance(init, VariableNode) or (
            hasattr(init, "__class__") and "ExpressionStatement" in str(init.__class__)
        ):
            return self.generate_statement(init, 0).strip().rstrip(";")
        return self.generate_expression(init).strip().rstrip(";")

    def generate_expression(self, expr, is_main=False):
        """Render a CrossGL AST expression into GLSL expression syntax."""
        if expr is None:
            return ""
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "VariableNode" in str(type(expr)):
            if hasattr(expr, "name"):
                if expr.name in self.current_resource_aliases:
                    return self.current_resource_aliases[expr.name]["expression"]
                if expr.name in getattr(self, "enum_variant_constants", {}):
                    return enum_value_expression(self, expr.name)
                return self.expression_identifier_name(expr.name)
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "IdentifierNode" in str(type(expr)):
            if expr.name in self.current_resource_aliases:
                return self.current_resource_aliases[expr.name]["expression"]
            if expr.name in getattr(self, "enum_variant_constants", {}):
                return enum_value_expression(self, expr.name)
            return self.expression_identifier_name(expr.name)
        elif hasattr(expr, "__class__") and "LiteralNode" in str(type(expr)):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if (
                literal_type == "uint"
                and isinstance(expr.value, int)
                and not isinstance(expr.value, bool)
            ):
                return f"{expr.value}u"
            if isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            return str(expr.value)
        elif hasattr(expr, "__class__") and "BinaryOpNode" in str(type(expr)):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = self.map_operator(expr.op)
            return f"({left} {op} {right})"
        elif hasattr(expr, "__class__") and "AssignmentNode" in str(type(expr)):
            return self.generate_assignment(expr)
        elif hasattr(expr, "__class__") and "UnaryOpNode" in str(type(expr)):
            operand = self.generate_expression(expr.operand)
            op = self.map_operator(expr.op)
            return f"({op}{operand})"
        elif isinstance(expr, WaveOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, RayTracingOpNode):
            args = [self.generate_expression(arg) for arg in expr.arguments]
            return self.map_ray_tracing_intrinsic(expr.operation, args)
        elif isinstance(expr, MeshOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            operation = self.map_mesh_intrinsic(expr.operation)
            return f"{operation}({args})"
        elif isinstance(expr, RayQueryOpNode):
            query = self.generate_expression(expr.query_expr)
            args = [self.generate_expression(arg) for arg in expr.arguments]
            return self.map_ray_query_intrinsic(expr.operation, query, args)
        elif hasattr(expr, "__class__") and "ArrayAccessNode" in str(type(expr)):
            # Handle array access properly
            if hasattr(expr, "array") and hasattr(expr, "index"):
                array = self.generate_expression(expr.array)
                index = self.generate_expression(expr.index)
                return f"{array}[{index}]"
            else:
                return str(expr)
        elif isinstance(expr, ConstructorNode):
            enum_constructor = generate_enum_constructor_expression(self, expr)
            if enum_constructor is not None:
                return enum_constructor
            constructor = generate_struct_constructor_expression(self, expr)
            if constructor is not None:
                return constructor
            return str(expr)
        elif hasattr(expr, "__class__") and "FunctionCallNode" in str(type(expr)):
            # Map function names to GLSL equivalents
            func_expr = getattr(expr, "function", getattr(expr, "name", expr))
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
            original_func_name = func_name

            ray_query_member_call = self.ray_query_member_function_call(
                func_expr, expr.args
            )
            if ray_query_member_call is not None:
                return ray_query_member_call

            synchronization_call = self.synchronization_function_call(
                original_func_name, expr.args
            )
            if synchronization_call is not None:
                return synchronization_call

            func_name = self.function_map.get(func_name, func_name)

            static_generic_call = generate_static_generic_numeric_call(
                self,
                original_func_name,
            )
            if static_generic_call is not None:
                return static_generic_call

            enum_constructor = generate_enum_constructor_call(
                self, original_func_name, expr.args
            )
            if enum_constructor is not None:
                return enum_constructor

            texture_call = self.generate_texture_call(func_name, expr.args)
            if texture_call is not None:
                return texture_call

            buffer_call = self.generate_buffer_call(func_name, expr.args)
            if buffer_call is not None:
                return buffer_call

            specialized_func_name = generic_function_call_name(
                self,
                original_func_name,
                expr.args,
            )
            if specialized_func_name is not None:
                func_name = specialized_func_name
                callee = specialized_func_name

            constructor = self.glsl_constructor_type(func_name)
            if constructor:
                args = ", ".join(
                    self.generate_expression_with_expected(arg, None)
                    for arg in expr.args
                )
                return f"{constructor}({args})"

            self.validate_function_image_access_arguments(func_name, expr.args)
            resource_specialization = self.glsl_resource_function_call_specialization(
                func_name,
                expr.args,
            )
            argument_func_name = original_func_name
            if resource_specialization is not None:
                func_name = resource_specialization.name
                callee = resource_specialization.name
                argument_func_name = func_name
                call_args = self.glsl_resource_specialized_call_arguments(
                    resource_specialization,
                    expr.args,
                )
                call_args = self.filter_sampler_arguments(func_name, call_args)
            else:
                call_args = self.filter_sampler_arguments(original_func_name, expr.args)
            args = ", ".join(
                self.generate_function_call_arguments(argument_func_name, call_args)
            )
            return f"{func_name or callee}({args})"
        elif hasattr(expr, "__class__") and "MemberAccessNode" in str(type(expr)):
            flattened_member = self.flattened_stage_member_name(expr)
            if flattened_member is not None:
                return flattened_member
            obj = self.generate_expression_with_expected(expr.object, None)
            return f"{obj}.{expr.member}"
        elif hasattr(expr, "__class__") and "TernaryOpNode" in str(type(expr)):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        else:
            return str(expr)

    def synchronization_function_call(self, func_name, args):
        if args or func_name in self.function_return_types:
            return None
        if func_name == "workgroupBarrier":
            return "barrier()"
        return None

    def ray_query_member_function_call(self, func_expr, args):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        query = self.generate_expression_with_expected(func_expr.object, None)
        generated_args = [self.generate_expression(arg) for arg in args]
        return self.map_ray_query_intrinsic(func_expr.member, query, generated_args)

    def generate_buffer_call(self, func_name, args):
        if func_name == "buffer_load" and len(args) >= 2:
            index = self.generate_expression(args[1])
            return f"{self.structured_buffer_access_expression(args[0], index)}"
        if func_name == "buffer_store" and len(args) >= 3:
            index = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            array_access = self.structured_buffer_array_parameter_access(args[0])
            if array_access is not None:
                info, selector = array_access
                return self.structured_buffer_array_select_expression(
                    info,
                    selector,
                    lambda name: f"({name}[{index}] = {value})",
                )
            return (
                f"{self.structured_buffer_access_expression(args[0], index)} = {value}"
            )
        if func_name == "buffer_append" and len(args) >= 2:
            value = self.generate_expression(args[1])
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return (
                    "/* unsupported GLSL buffer append: requires append/consume "
                    "counter buffer */"
                )
            index = f"atomicAdd({counter}, 1u)"
            return (
                f"{self.structured_buffer_access_expression(args[0], index)} = {value}"
            )
        if func_name == "buffer_consume" and args:
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return (
                    "0 /* unsupported GLSL buffer consume: requires append/consume "
                    "counter buffer */"
                )
            index = f"(atomicAdd({counter}, uint(-1)) - 1u)"
            return self.structured_buffer_access_expression(args[0], index)
        if func_name == "buffer_dimensions" and args:
            length_expr = self.structured_buffer_length_expression(args[0])
            if len(args) >= 2:
                target = self.generate_expression(args[1])
                return f"{target} = {length_expr}"
            return length_expr
        return None

    def structured_buffer_access_expression(self, buffer_arg, element_index):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            return self.structured_buffer_array_select_expression(
                info,
                selector,
                lambda name: f"{name}[{element_index}]",
            )
        buffer = self.generate_expression(buffer_arg)
        buffer_name = self.expression_name(buffer_arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{buffer}.{instance_member}[{element_index}]"
        return f"{buffer}[{element_index}]"

    def structured_buffer_length_expression(self, buffer_arg):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            return self.structured_buffer_array_select_expression(
                info,
                selector,
                lambda name: f"{name}.length()",
            )
        buffer = self.generate_expression(buffer_arg)
        buffer_name = self.expression_name(buffer_arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{buffer}.{instance_member}.length()"
        return f"{buffer}.length()"

    def structured_buffer_counter_reference(self, buffer_arg):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            counter_names = info.get("counter_expanded_names")
            if counter_names:
                counter_info = {**info, "expanded_names": counter_names}
                return self.structured_buffer_array_select_expression(
                    counter_info,
                    selector,
                    lambda name: f"{name}[0]",
                )

        counter_data = self.structured_buffer_counter_data_argument(buffer_arg)
        if counter_data is None:
            return None
        return f"{counter_data}[0]"

    def flattened_stage_member_name(self, expr):
        object_name = self.expression_name(getattr(expr, "object", None))
        if object_name in self.current_stage_inputs:
            return self.current_stage_inputs[object_name].get(expr.member)
        if object_name in self.current_stage_outputs:
            return self.current_stage_outputs[object_name].get(expr.member)
        return None

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
        return None

    def collect_function_parameter_types(self, functions):
        parameter_types = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            types = []
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                types.append(self.type_name_string(raw_type))
            parameter_types[func_name] = types
        return parameter_types

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

    def function_call_argument_pairs(self, func_name, call_args):
        parameter_infos = list(self.function_parameter_infos.get(func_name, []))
        if not parameter_infos:
            parameter_infos = [
                (None, param_type)
                for param_type in self.function_parameter_types.get(func_name, [])
            ]
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        filtered_parameter_infos = [
            param_info
            for index, param_info in enumerate(parameter_infos)
            if index not in skipped_indices
        ]
        pairs = []
        for index, arg in enumerate(call_args):
            param_name, param_type = (
                filtered_parameter_infos[index]
                if index < len(filtered_parameter_infos)
                else (None, None)
            )
            pairs.append((arg, param_name, param_type))
        return pairs

    def generate_function_call_arguments(self, func_name, call_args):
        args = []
        for arg, param_name, param_type in self.function_call_argument_pairs(
            func_name, call_args
        ):
            generated = self.generate_function_call_argument(
                func_name, arg, param_name, param_type
            )
            if isinstance(generated, list):
                args.extend(generated)
            else:
                args.append(generated)
        return args

    def generate_function_call_argument(self, func_name, arg, param_name, param_type):
        buffer_array = self.structured_buffer_array_parameter_info(
            param_type, param_name, func_name
        )
        if buffer_array is not None:
            generated_args = self.structured_buffer_array_data_arguments(
                arg, buffer_array["count"]
            )
            if self.structured_buffer_requires_counter(buffer_array["base_type"]):
                generated_args += self.structured_buffer_array_counter_arguments(
                    arg, buffer_array["count"]
                )
            return generated_args
        if (
            param_type is not None
            and self.is_structured_buffer_type(param_type)
            and not self.is_array_type(param_type)
        ):
            data_arg = self.structured_buffer_data_argument(arg)
            if self.structured_buffer_requires_counter(param_type):
                return [data_arg, self.structured_buffer_counter_data_argument(arg)]
            return data_arg
        return self.generate_expression(arg)

    def structured_buffer_array_data_arguments(self, arg, count):
        arg_name = self.expression_name(arg)
        expanded = self.current_structured_buffer_array_parameters.get(arg_name)
        if expanded is not None:
            expanded_names = expanded["expanded_names"]
            return expanded_names[:count]

        arg_code = self.generate_expression(arg)
        instance_member = self.structured_buffer_instance_members.get(arg_name)
        if instance_member:
            return [f"{arg_code}[{index}].{instance_member}" for index in range(count)]
        return [f"{arg_code}[{index}]" for index in range(count)]

    def structured_buffer_data_argument(self, arg):
        arg_code = self.generate_expression(arg)
        buffer_name = self.expression_name(arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{arg_code}.{instance_member}"
        return arg_code

    def structured_buffer_array_counter_arguments(self, arg, count):
        arg_name = self.expression_name(arg)
        expanded = self.current_structured_buffer_array_parameters.get(arg_name)
        if expanded is not None:
            counter_names = expanded.get("counter_expanded_names", [])
            return counter_names[:count]

        instance_name = self.structured_buffer_counter_instances.get(arg_name)
        member_name = self.structured_buffer_counter_members.get(arg_name)
        if instance_name and member_name:
            return [f"{instance_name}[{index}].{member_name}" for index in range(count)]
        return [self.structured_buffer_counter_data_argument(arg)] * count

    def structured_buffer_counter_data_argument(self, arg):
        array_access = self.structured_buffer_array_parameter_access(arg)
        if array_access is not None:
            info, selector = array_access
            counter_names = info.get("counter_expanded_names")
            if counter_names:
                counter_info = {**info, "expanded_names": counter_names}
                return self.structured_buffer_array_select_expression(
                    counter_info,
                    selector,
                    lambda name: name,
                )

        arg_name = self.expression_name(arg)
        counter_parameter = self.current_structured_buffer_counter_parameters.get(
            arg_name
        )
        if counter_parameter is not None:
            return counter_parameter

        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_name = self.expression_name(array_expr)
            instance_name = self.structured_buffer_counter_instances.get(array_name)
            member_name = self.structured_buffer_counter_members.get(array_name)
            if instance_name and member_name:
                index = self.generate_expression(index_expr)
                return f"{instance_name}[{index}].{member_name}"

        member_name = self.structured_buffer_counter_members.get(arg_name)
        if member_name and arg_name not in self.structured_buffer_counter_instances:
            return member_name
        return None

    def structured_buffer_array_parameter_access(self, buffer_arg):
        if not isinstance(buffer_arg, ArrayAccessNode) and not (
            hasattr(buffer_arg, "__class__")
            and "ArrayAccess" in str(buffer_arg.__class__)
        ):
            return None
        array_expr = getattr(
            buffer_arg, "array", getattr(buffer_arg, "array_expr", None)
        )
        array_name = self.expression_name(array_expr)
        info = self.current_structured_buffer_array_parameters.get(array_name)
        if info is None:
            return None
        selector = getattr(buffer_arg, "index", getattr(buffer_arg, "index_expr", None))
        return info, selector

    def structured_buffer_array_select_expression(self, info, selector, branch):
        expanded_names = info["expanded_names"]
        literal_index = self.literal_int_value(selector, self.literal_int_constants)
        if literal_index is not None and 0 <= literal_index < len(expanded_names):
            return branch(expanded_names[literal_index])

        selector_code = self.generate_expression(selector)
        selector_type = self.map_type(self.expression_result_type(selector))

        def selector_literal(index):
            return f"{index}u" if selector_type == "uint" else str(index)

        expression = branch(expanded_names[-1])
        for index in range(len(expanded_names) - 2, -1, -1):
            expression = (
                f"(({selector_code} == {selector_literal(index)}) ? "
                f"{branch(expanded_names[index])} : {expression})"
            )
        return expression

    def is_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return True
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        return "[" in type_string and "]" in type_string

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
        if (
            sampler_name in self.sampler_variables
            or sampler_name in self.current_sampler_parameters
        ):
            return True
        arg_type = self.expression_result_type(args[1])
        return arg_type is not None and self.is_sampler_type(arg_type)

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, coord, extra_args

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        alias = self.current_resource_aliases.get(texture_name)
        if alias is not None:
            return alias.get("type")
        return self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        return self.expression_result_type(texture_arg)

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is not None and self.is_inferable_resource_array_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"OpenGL texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def validate_image_resource_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_type(texture_type):
            return
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"OpenGL image operation '{func_name}' requires a storage "
            f"image resource argument: {texture_name}"
        )

    def validate_image_access_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        access = self.image_resource_access(args[0])
        if access is None or access == "read_write":
            return
        texture_name = expression_debug_name(args[0])
        if func_name == "imageLoad" and access == "write":
            raise ValueError(
                f"OpenGL image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got writeonly"
            )
        if func_name == "imageStore" and access == "read":
            raise ValueError(
                f"OpenGL image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got readonly"
            )
        if is_image_atomic_operation(func_name):
            access_name = "readonly" if access == "read" else "writeonly"
            raise ValueError(
                f"OpenGL image operation '{func_name}' requires read-write "
                f"storage image access for {texture_name}: got {access_name}"
            )

    def image_atomic_value_arguments(self, func_name, args):
        image_type = self.texture_argument_resource_type(args[0])
        has_sample = self.is_multisample_storage_image_type(image_type)
        return shared_image_atomic_value_arguments(func_name, args, has_sample)

    def scalar_expression_kind(self, expr):
        return numeric_scalar_expression_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_value_argument_types(
        self, func_name, args, component_kind, image_format
    ):
        mismatch = image_atomic_value_kind_mismatch(
            func_name,
            self.image_atomic_value_arguments(func_name, args),
            component_kind,
            self.scalar_expression_kind,
        )
        if mismatch is None:
            return
        value_arg, value_kind = mismatch
        raise ValueError(
            image_atomic_value_kind_error(
                "OpenGL",
                func_name,
                image_format,
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
        self, func_name, component_kind, image_format
    ):
        expected_kind = image_atomic_result_kind_mismatch(
            self.scalar_expected_kind(), component_kind
        )
        if expected_kind is None:
            return
        raise ValueError(
            image_atomic_result_kind_error(
                "OpenGL", func_name, image_format, component_kind, expected_kind
            )
        )

    def validate_image_atomic_format_argument(self, func_name, args):
        if not is_image_atomic_operation(func_name) or not args:
            return
        image_format = self.image_resource_format(args[0])
        if image_format is None:
            return
        component_kind = image_atomic_explicit_format_component_kind(
            func_name, image_format
        )
        if component_kind is not None:
            self.validate_image_atomic_value_argument_types(
                func_name,
                args,
                component_kind,
                image_format,
            )
            self.validate_image_atomic_result_type(
                func_name,
                component_kind,
                image_format,
            )
            return
        raise ValueError(image_atomic_format_error("OpenGL", func_name, image_format))

    def validate_function_image_access_arguments(self, func_name, args):
        callee_requirements = self.function_image_access_requirements.get(func_name)
        if not callee_requirements:
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.image_resource_access(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            raise ValueError(
                f"OpenGL function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got {actual_label}"
            )

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        return is_integer_coordinate_type_name(base_type, mapped_type)

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = self.is_multisample_texture_resource_type(texture_type)
        is_storage_image = self.is_storage_image_type(texture_type)

        coordinate_dimension = None
        if texture_type and "Cube" not in texture_type:
            for prefix in ("iimage", "uimage", "image"):
                if texture_type.startswith(f"{prefix}2DMSArray"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}2DMS"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}2DArray"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}3D"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}2D"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}1DArray"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}1D"):
                    coordinate_dimension = 1
                    break
            if coordinate_dimension is None:
                coordinate_dimension = {
                    "sampler1D": 1,
                    "sampler1DArray": 2,
                    "sampler2D": 2,
                    "sampler2DArray": 3,
                    "sampler2DMS": 2,
                    "sampler2DMSArray": 3,
                    "sampler3D": 3,
                    "isampler1D": 1,
                    "isampler1DArray": 2,
                    "isampler2D": 2,
                    "isampler2DArray": 3,
                    "isampler2DMS": 2,
                    "isampler2DMSArray": 3,
                    "isampler3D": 3,
                    "usampler1D": 1,
                    "usampler1DArray": 2,
                    "usampler2D": 2,
                    "usampler2DArray": 3,
                    "usampler2DMS": 2,
                    "usampler2DMSArray": 3,
                    "usampler3D": 3,
                }.get(texture_type)

        offset_dimension = None
        if texture_type and "Cube" not in texture_type:
            offset_dimension = {
                "sampler1D": 1,
                "sampler1DArray": 1,
                "sampler2D": 2,
                "sampler2DArray": 2,
                "sampler2DShadow": 2,
                "sampler2DArrayShadow": 2,
                "sampler3D": 3,
                "isampler1D": 1,
                "isampler1DArray": 1,
                "isampler2D": 2,
                "isampler2DArray": 2,
                "isampler3D": 3,
                "usampler1D": 1,
                "usampler1DArray": 1,
                "usampler2D": 2,
                "usampler2DArray": 2,
                "usampler3D": 3,
            }.get(texture_type)

        gradient_dimension = None
        if texture_type and not is_multisample and not is_storage_image:
            if "Cube" in texture_type:
                gradient_dimension = 3
            else:
                gradient_dimension = {
                    "sampler1D": 1,
                    "sampler1DArray": 1,
                    "sampler2D": 2,
                    "sampler2DArray": 2,
                    "sampler2DShadow": 2,
                    "sampler2DArrayShadow": 2,
                    "sampler3D": 3,
                    "isampler1D": 1,
                    "isampler1DArray": 1,
                    "isampler2D": 2,
                    "isampler2DArray": 2,
                    "isampler3D": 3,
                    "usampler1D": 1,
                    "usampler1DArray": 1,
                    "usampler2D": 2,
                    "usampler2DArray": 2,
                    "usampler3D": 3,
                }.get(texture_type)

        query_lod_coordinate_dimension = None
        if texture_type and not is_multisample and not is_storage_image:
            query_lod_coordinate_dimension = {
                "sampler1D": 1,
                "sampler1DArray": 2,
                "sampler2D": 2,
                "sampler2DArray": 3,
                "sampler2DShadow": 2,
                "sampler2DArrayShadow": 3,
                "sampler3D": 3,
                "samplerCube": 3,
                "samplerCubeArray": 4,
                "samplerCubeShadow": 3,
                "samplerCubeArrayShadow": 4,
                "isampler1D": 1,
                "isampler1DArray": 2,
                "isampler2D": 2,
                "isampler2DArray": 3,
                "isampler3D": 3,
                "isamplerCube": 3,
                "isamplerCubeArray": 4,
                "usampler1D": 1,
                "usampler1DArray": 2,
                "usampler2D": 2,
                "usampler2DArray": 3,
                "usampler3D": 3,
                "usamplerCube": 3,
                "usamplerCubeArray": 4,
            }.get(texture_type)

        return texture_resource_dimension_descriptor(
            texture_type,
            sampling,
            coordinate_dimension=coordinate_dimension,
            offset_dimension=offset_dimension,
            gradient_dimension=gradient_dimension,
            query_lod_coordinate_dimension=query_lod_coordinate_dimension,
            is_multisample=is_multisample,
        )

    def resource_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)["coordinate_dimension"]

    def resource_offset_dimension(self, func_name, texture_type):
        descriptor = self.texture_dimension_descriptor(texture_type)
        return descriptor[texture_resource_offset_dimension_key(func_name)]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

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
                "OpenGL",
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
                "OpenGL",
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
                        "OpenGL",
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
                    "OpenGL",
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
                    "OpenGL",
                    func_name,
                    expression_debug_name(args[coord_index]),
                    self.type_name_string(coord_type),
                )
            )
        if coord_dimension == expected_dimension:
            return
        raise ValueError(
            texture_query_lod_coordinate_dimension_error(
                "OpenGL",
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
                        "OpenGL",
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
                    "OpenGL",
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
        return shared_texture_argument_diagnostic_type(
            arg,
            self.texture_resource_type,
            self.expression_name,
            self.expression_result_type,
            self.sampler_variables | self.current_sampler_parameters,
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
                "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
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
        if not self.is_multisample_texture_resource_type(texture_type):
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        if sample_type is None or self.is_scalar_integer_type(sample_type):
            return
        raise ValueError(
            texture_multisample_sample_type_error(
                "OpenGL",
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
            self.is_multisample_storage_image_type(texture_type),
            "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
                "texture gather",
                func_name,
                "a scalar integer",
                "component",
                expression_debug_name(args[component_index]),
                self.type_name_string(component_type),
            )
        )

    def validate_texture_call_arity(self, func_name, args):
        validate_texture_operation_arity(
            "OpenGL",
            func_name,
            args,
            self.texture_resource_operation_names(),
            self.texture_call_uses_explicit_sampler,
        )

    def texture_resource_operation_names(self):
        return texture_image_resource_operation_names(IMAGE_RESOURCE_INTRINSIC_NAMES)

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("GLSL", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

    def is_array_expression(self, node):
        type_name = self.type_name_string(self.expression_result_type(node))
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

    def texture_gather_component_value(self, component_arg):
        if component_arg is None:
            return None
        return self.literal_int_value(component_arg, self.literal_int_constants)

    def texture_gather_call_expression(
        self, function_name, texture_name, coord, offset_arg=None, component=None
    ):
        args = [texture_name, coord]
        if offset_arg is not None:
            args.append(offset_arg)
        if component is not None:
            args.append(str(component))
        return f"{function_name}({', '.join(args)})"

    def texture_gather_offsets_expression(
        self, texture_name, coord, offset_args, component
    ):
        component_suffixes = ("x", "y", "z", "w")
        component_values = []
        for index, offset_arg in enumerate(offset_args):
            gather = self.texture_gather_call_expression(
                "textureGatherOffset",
                texture_name,
                coord,
                self.generate_expression(offset_arg),
                component,
            )
            component_values.append(f"{gather}.{component_suffixes[index]}")
        return f"vec4({', '.join(component_values)})"

    def texture_gather_dynamic_component_expression(self, build_expression, component):
        component_calls = [build_expression(index) for index in range(4)]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return unsupported_texture_gather_call_expression("GLSL", func_name, reason)

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        gather_types = {
            "sampler2D",
            "sampler2DArray",
            "samplerCube",
            "samplerCubeArray",
        }
        gather_offset_types = {"sampler2D", "sampler2DArray"}
        sample_offset_types = {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler3D",
            "sampler2DArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
        }
        compare_offset_types = {"sampler2DShadow", "sampler2DArrayShadow"}
        compare_lod_types = {"sampler2DShadow"}
        compare_grad_types = {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
        }
        return {
            "texture_type": texture_type,
            "gather": texture_type in gather_types,
            "gather_offset": texture_type in gather_offset_types,
            "sample_offset": texture_type in sample_offset_types,
            "compare_offset": texture_type in compare_offset_types,
            "compare_lod": texture_type in compare_lod_types,
            "compare_grad": texture_type in compare_grad_types,
            "compare_lod_offset": texture_type in compare_lod_types,
            "compare_grad_offset": texture_type in compare_offset_types,
            "gather_compare_offset": texture_type in compare_offset_types,
        }

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_gather_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def unsupported_texture_projected_call(self, func_name, reason):
        return unsupported_projected_texture_call_expression("GLSL", func_name, reason)

    def projected_cube_texture_coordinate(self, texture_type, coord_arg, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type != "samplerCube":
            return None

        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        if coord_type not in {"vec4", "float4"}:
            return None
        return (
            f"{self.vector_component(coord, 'xyz')} / "
            f"{self.vector_component(coord, 'w')}"
        )

    def generate_projected_cube_texture_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_projected_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        texture_type = self.texture_resource_type(args[0])
        projected_coord = self.projected_cube_texture_coordinate(
            texture_type, args[coord_index], coord
        )
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name,
                "requires 1D, 2D, 2D-array, 3D, or cube projection coordinates",
            )

        if is_projected_texture_basic_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            mapped_args = [texture_name, projected_coord]
            if extra_args:
                mapped_args.append(self.generate_expression(extra_args[0]))
            return f"texture({', '.join(mapped_args)})"

        if is_projected_texture_lod_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            lod = self.generate_expression(extra_args[0])
            return f"textureLod({texture_name}, {projected_coord}, {lod})"

        if is_projected_texture_grad_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return f"textureGrad({texture_name}, {projected_coord}, {ddx}, {ddy})"

        if (
            is_projected_texture_basic_offset_operation(func_name)
            or is_projected_texture_lod_offset_operation(func_name)
            or is_projected_texture_grad_offset_operation(func_name)
        ):
            return self.unsupported_texture_projected_call(
                func_name, texture_sample_offset_capability_error("GLSL")
            )

        return self.unsupported_texture_projected_call(
            func_name, unsupported_projected_texture_operation_error()
        )

    def texture_sample_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        return unsupported_texture_offset_call_expression("GLSL", func_name, reason)

    def is_multisample_texture_resource_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "sampler2DMS",
            "sampler2DMSArray",
            "image2DMS",
            "image2DMSArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }

    def is_multisample_storage_image_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "image2DMS",
            "image2DMSArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_call_vector_expression(
            "GLSL", func_name, texture_type
        )

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_compare_scalar_expression(
            "GLSL", func_name, texture_type
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_gather_compare_vector_expression(
            "GLSL", func_name, texture_type
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_query_lod_expression(
            "GLSL", texture_type
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_levels_expression("GLSL", texture_type)

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_lod_expression("GLSL", texture_type)

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_type(texture_type):
            return None

        texture_type = self.resource_base_type(texture_type)
        if is_storage_image_texture_comparison_operation(func_name):
            return unsupported_storage_image_texture_comparison_scalar_expression(
                "GLSL", func_name, texture_type
            )

        if is_storage_image_texture_operation(func_name):
            return unsupported_storage_image_texture_operation_vector_expression(
                "GLSL", func_name, texture_type
            )

        return None

    def unsupported_texture_samples_query_call(self):
        return unsupported_texture_samples_query_call_expression("GLSL")

    def image_size_expression(self, image_arg):
        image_name = self.generate_expression(image_arg)
        return f"imageSize({image_name})"

    def texture_query_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if not texture_type:
            return None
        return {
            "function_name": (
                "imageSize"
                if self.is_storage_image_type(texture_type)
                else "textureSize"
            ),
        }

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        return {
            "texture_type": texture_type,
            "storage_image": self.is_storage_image_type(texture_type),
            "multisample": self.is_multisample_texture_resource_type(texture_type),
            "size_descriptor": self.texture_query_size_descriptor(texture_type),
        }

    def texture_query_levels_expression(self, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        if descriptor["multisample"]:
            return texture_query_levels_multisample_expression()
        return None

    def texture_samples_expression(self, func_name, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        if descriptor["texture_type"] and not descriptor["multisample"]:
            return self.unsupported_texture_samples_query_call()
        if (
            not is_texture_samples_query_operation(func_name)
            and descriptor["multisample"]
        ):
            texture_name = self.generate_expression(texture_arg)
            return texture_samples_query_expression("GLSL", texture_name)
        return None

    def texture_size_query_expression(self, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        size_descriptor = descriptor["size_descriptor"]
        if size_descriptor and size_descriptor["function_name"] == "imageSize":
            return self.image_size_expression(texture_arg)
        return None

    def unsupported_multisample_texel_fetch_offset_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texel_fetch_offset_expression(
            "GLSL", texture_type
        )

    def is_cube_texture_resource_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "samplerCube",
            "samplerCubeArray",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_cube_texel_fetch_expression("GLSL", func_name, texture_type)

    def generate_texture_gather_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)
        if is_texture_gather_basic_operation(
            func_name
        ) and not self.texture_gather_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_capability_error()
            )
        if is_texture_gather_offset_operation(
            func_name
        ) and not self.texture_gather_offset_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_offset_capability_error()
            )

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
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        elif is_texture_gather_multi_offset_operation(func_name):
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offsets_argument_count_error()
                )
        else:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_operation_error()
            )

        component = self.texture_gather_component_value(component_arg)
        if component is not None:
            if component not in {0, 1, 2, 3}:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_component_literal_error()
                )
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, component
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if is_texture_gather_single_offset_operation(func_name)
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, component
            )

        if component_arg is None:
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, None
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if is_texture_gather_single_offset_operation(func_name)
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg
            )

        component_expr = self.generate_expression(component_arg)
        if is_texture_gather_multi_offset_operation(func_name):
            return self.texture_gather_dynamic_component_expression(
                lambda option: self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, option
                ),
                component_expr,
            )

        offset_arg = self.generate_expression(offset_args[0]) if offset_args else None
        function_name = (
            "textureGatherOffset"
            if is_texture_gather_single_offset_operation(func_name)
            else "textureGather"
        )
        return self.texture_gather_dynamic_component_expression(
            lambda option: self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, option
            ),
            component_expr,
        )

    def texture_compare_coordinate(self, texture_type, coord, compare):
        texture_type = self.resource_base_type(texture_type)
        if texture_type == "samplerCubeArrayShadow":
            return None
        constructor = (
            "vec4"
            if texture_type
            in {
                "sampler2DArrayShadow",
                "samplerCubeShadow",
            }
            else "vec3"
        )
        return f"{constructor}({coord}, {compare})"

    def texture_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

    def texture_compare_lod_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_lod"]

    def texture_compare_grad_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_grad"]

    def texture_compare_lod_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_lod_offset"]

    def texture_compare_grad_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_grad_offset"]

    def texture_compare_projected_coordinate(
        self, texture_type, coord_arg, coord, compare
    ):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "sampler2DShadow":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            projected_coord = f"{self.vector_component(coord, 'xy')} / {divisor}"
            return f"vec3({projected_coord}, {compare})"

        if texture_type == "samplerCubeShadow":
            if coord_type not in {"vec4", "float4"}:
                return None
            projected_coord = (
                f"{self.vector_component(coord, 'xyz')} / "
                f"{self.vector_component(coord, 'w')}"
            )
            return f"vec4({projected_coord}, {compare})"

        if texture_type != "sampler2DArrayShadow" or coord_type not in {
            "vec4",
            "float4",
        }:
            return None

        projected_coord = (
            f"{self.vector_component(coord, 'xy')} / "
            f"{self.vector_component(coord, 'w')}"
        )
        layer = self.vector_component(coord, "z")
        return f"vec4({projected_coord}, {layer}, {compare})"

    def unsupported_texture_compare_call(self, func_name, reason):
        return unsupported_texture_compare_scalar_expression("GLSL", func_name, reason)

    def generate_texture_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, texture_compare_argument_error()
            )

        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if is_projected_texture_compare_operation(func_name):
            coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
            compare_coord = self.texture_compare_projected_coordinate(
                texture_type, args[coord_index], coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_projected_coordinate_error("GLSL")
                )

            if is_texture_compare_basic_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return f"texture({texture_name}, {compare_coord})"

            if is_texture_compare_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("GLSL")
                    )
                offset = self.generate_expression(extra_args[1])
                return f"textureOffset({texture_name}, {compare_coord}, {offset})"

            if is_texture_compare_lod_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_projected_lod_array_error()
                    )
                if not self.texture_compare_lod_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, "explicit LOD requires 2D shadow samplers"
                    )
                lod = self.generate_expression(extra_args[1])
                return f"textureLod({texture_name}, {compare_coord}, {lod})"

            if is_texture_compare_lod_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_projected_lod_array_error()
                    )
                if not self.texture_compare_lod_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, "explicit LOD offsets require 2D shadow samplers"
                    )
                lod = self.generate_expression(extra_args[1])
                offset = self.generate_expression(extra_args[2])
                return (
                    f"textureLodOffset({texture_name}, {compare_coord}, "
                    f"{lod}, {offset})"
                )

            if is_texture_compare_grad_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_grad_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "explicit gradients require 2D, 2D-array, or cube shadow samplers",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

            if is_texture_compare_grad_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_grad_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "explicit gradient offsets require 2D or 2D-array shadow samplers",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                offset = self.generate_expression(extra_args[3])
                return (
                    f"textureGradOffset({texture_name}, {compare_coord}, "
                    f"{ddx}, {ddy}, {offset})"
                )

            return self.unsupported_texture_compare_call(
                func_name, unsupported_texture_compare_operation_error(projected=True)
            )

        if is_texture_compare_basic_operation(func_name):
            if texture_type == "samplerCubeArrayShadow":
                return f"texture({texture_name}, {coord}, {compare})"
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            return f"texture({texture_name}, {compare_coord})"

        if is_texture_compare_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("GLSL")
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            offset = self.generate_expression(extra_args[1])
            return f"textureOffset({texture_name}, {compare_coord}, {offset})"

        if is_texture_compare_lod_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_lod_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD requires 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            lod = self.generate_expression(extra_args[1])
            return f"textureLod({texture_name}, {compare_coord}, {lod})"

        if is_texture_compare_lod_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_lod_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD offsets require 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            lod = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            return f"textureLodOffset({texture_name}, {compare_coord}, {lod}, {offset})"

        if is_texture_compare_grad_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_grad_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradients require 2D, 2D-array, or cube shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

        if is_texture_compare_grad_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_grad_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradient offsets require 2D or 2D-array shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset = self.generate_expression(extra_args[3])
            return (
                f"textureGradOffset({texture_name}, {compare_coord}, "
                f"{ddx}, {ddy}, {offset})"
            )

        return None

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return unsupported_texture_gather_compare_call_expression(
            "GLSL", func_name, reason
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

    def generate_texture_gather_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_argument_error()
            )

        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if func_name == "textureGatherCompare":
            count_error = texture_gather_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_gather_compare_call(
                    func_name, count_error
                )
            return f"textureGather({texture_name}, {coord}, {compare})"

        count_error = texture_gather_compare_extra_argument_count_error(
            func_name, len(extra_args)
        )
        if count_error:
            return self.unsupported_texture_gather_compare_call(func_name, count_error)
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("GLSL")
            )
        offset = self.generate_expression(extra_args[1])
        return f"textureGatherOffset({texture_name}, {coord}, {compare}, {offset})"

    def image_resource_format(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )

    def image_resource_access(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_access_parameters,
            self.image_variable_accesses,
        )

    def image_atomic_parameter_requires_diagnostic(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name is None:
            return False
        if texture_name in self.current_resource_aliases:
            return False
        return (
            texture_name in self.current_texture_parameters
            and texture_name not in self.texture_variable_types
        )

    def image_atomic_zero_value(self, texture_type, image_format):
        component_kind = image_format_component_kind(image_format)
        if component_kind is None:
            texture_type = self.resource_base_type(texture_type)
            if texture_type and texture_type.startswith("uimage"):
                component_kind = "uint"
            elif texture_type and texture_type.startswith("image"):
                component_kind = "float"
            else:
                component_kind = "int"
        return storage_image_atomic_zero_value(component_kind)

    def image_atomic_parameter_fallback_call(self, func_name, args):
        texture_type = self.texture_argument_resource_type(args[0])
        image_format = self.image_resource_format(args[0])
        zero_value = self.image_atomic_zero_value(texture_type, image_format)
        texture_type = self.resource_base_type(texture_type) or expression_debug_name(
            args[0]
        )
        return image_atomic_diagnostic_expression(
            "GLSL",
            func_name,
            f"{texture_type} parameter",
            zero_value,
        )

    def generate_image_atomic_call(self, func_name, args):
        if self.image_atomic_parameter_requires_diagnostic(args[0]):
            return self.image_atomic_parameter_fallback_call(func_name, args)
        return "{}({})".format(
            func_name,
            ", ".join(self.generate_expression(arg) for arg in args),
        )

    def is_integer_image_type(self, texture_type):
        return is_glsl_integer_image_type(texture_type)

    def is_scalar_image_format(self, image_format):
        return is_scalar_image_format(image_format)

    def is_two_component_image_format(self, image_format):
        return is_two_component_image_format(image_format)

    def is_scalar_integer_image_resource(self, texture_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(texture_type)

    def is_float_image_resource(self, texture_type):
        return is_glsl_float_image_resource(texture_type)

    def image_store_constructors_by_kind(self):
        return storage_image_store_constructors("vec4", "ivec4", "uvec4")

    def image_store_zero_values_by_kind(self):
        return storage_image_zero_values()

    def image_load_component_suffix(self, texture_type, image_format):
        return storage_image_load_component_suffix(
            image_format,
            expected_scalar=self.is_scalar_value_type(
                self.current_expression_expected_type
            ),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                texture_type, image_format
            ),
            float_resource=self.is_float_image_resource(texture_type),
        )

    def image_format_store_constructor(self, image_format):
        return storage_image_format_store_constructor(
            image_format, self.image_store_constructors_by_kind()
        )

    def integer_image_store_constructor(self, texture_type):
        if texture_type in {
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
            "iimage2DMS",
            "iimage2DMSArray",
        }:
            return "ivec4"
        if texture_type in {
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }:
            return "uvec4"
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
        self, texture_type, image_format, value, value_type=None
    ):
        return storage_image_store_value_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                texture_type, image_format
            ),
            float_resource=self.is_float_image_resource(texture_type),
            integer_constructor=self.integer_image_store_constructor(texture_type),
            float_constructor="vec4",
            constructors_by_kind=self.image_store_constructors_by_kind(),
            zero_values_by_kind=self.image_store_zero_values_by_kind(),
        )

    def glsl_storage_image_component_kind(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("uimage"):
            return "uint"
        if texture_type.startswith("iimage"):
            return "int"
        if texture_type.startswith("image"):
            return "float"
        return None

    def image_store_expected_value_type(
        self, texture_type, image_format, value_arg=None
    ):
        if value_arg is not None and self.is_vector_value_type(
            self.expression_result_type(value_arg)
        ):
            return None
        if image_format:
            return image_format_component_kind(image_format)
        component_kind = self.glsl_storage_image_component_kind(texture_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return None

    def image_load_result_type(self, image_arg):
        image_format = self.image_resource_format(image_arg)
        if image_format:
            return image_format_result_type(
                image_format,
                vector_prefixes={"float": "vec", "int": "ivec", "uint": "uvec"},
            )
        image_type = self.texture_argument_resource_type(image_arg)
        component_kind = self.glsl_storage_image_component_kind(image_type)
        if component_kind == "float":
            return "vec4"
        if component_kind == "int":
            return "ivec4"
        if component_kind == "uint":
            return "uvec4"
        return None

    def image_load_component_kind(self, texture_type, image_format):
        if image_format:
            return image_format_component_kind(image_format)
        return self.glsl_storage_image_component_kind(texture_type)

    def image_load_channel_count(self, texture_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                self.glsl_storage_image_component_kind(texture_type)
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
            scalar_types={"float", "double", "int", "uint", "bool"},
            excluded_type_markers=("mat",),
        )

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={"float", "double", "int", "uint", "bool"},
            excluded_type_markers=("mat",),
        )

    def image_store_channel_count(self, texture_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                self.glsl_storage_image_component_kind(texture_type)
            ),
        )

    def validate_image_store_value_shape(self, texture_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(texture_type, image_format)
        value_channels = image_store_value_shape_mismatch(
            expected_channels, self.expression_component_count(value_arg)
        )
        if value_channels is None:
            return
        format_label = image_format or self.resource_base_type(texture_type)
        raise ValueError(
            image_store_value_shape_error(
                "OpenGL",
                format_label,
                expression_debug_name(value_arg),
                expected_channels,
                value_channels,
            )
        )

    def validate_image_store_value_type(self, texture_type, image_format, value_arg):
        self.validate_image_store_value_shape(texture_type, image_format, value_arg)
        expected_kind = self.image_store_expected_value_type(texture_type, image_format)
        value_kind = image_store_value_kind_mismatch(
            expected_kind, self.expression_component_kind(value_arg)
        )
        if value_kind is None:
            return
        format_label = image_format or self.resource_base_type(texture_type)
        raise ValueError(
            image_store_value_kind_error(
                "OpenGL",
                format_label,
                expression_debug_name(value_arg),
                expected_kind,
                value_kind,
            )
        )

    def validate_image_load_result_type(self, texture_type, image_format):
        expected_kind = self.expected_component_kind()
        component_kind = self.image_load_component_kind(texture_type, image_format)
        format_label = image_format or self.resource_base_type(texture_type)
        if not should_validate_image_load_result_shape(expected_kind, component_kind):
            return
        expected_kind = image_load_result_kind_mismatch(expected_kind, component_kind)
        if expected_kind is not None:
            raise ValueError(
                image_load_result_kind_error(
                    "OpenGL", format_label, component_kind, expected_kind
                )
            )
        expected_channels = self.expected_component_count()
        loaded_channels = self.image_load_channel_count(texture_type, image_format)
        expected_channels = image_load_result_shape_mismatch(
            loaded_channels,
            expected_channels,
        )
        if expected_channels is None:
            return
        raise ValueError(
            image_load_result_shape_error(
                "OpenGL", format_label, loaded_channels, expected_channels
            )
        )

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

        if is_texture_query_levels_operation(func_name) and args:
            return self.texture_query_levels_expression(args[0])

        if is_resource_samples_query_operation(func_name) and args:
            return self.texture_samples_expression(func_name, args[0])

        if is_resource_size_query_operation(func_name) and args:
            return self.texture_size_query_expression(args[0])

        if len(args) < 2:
            return None

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 3 and self.is_multisample_storage_image_type(texture_type):
                sample = self.generate_expression(args[2])
                load_expr = f"imageLoad({image_name}, {coord}, {sample})"
            else:
                load_expr = f"imageLoad({image_name}, {coord})"
            image_format = self.image_resource_format(args[0])
            self.validate_image_load_result_type(texture_type, image_format)
            return f"{load_expr}{self.image_load_component_suffix(texture_type, image_format)}"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 4 and self.is_multisample_storage_image_type(texture_type):
                sample = self.generate_expression(args[2])
                value_arg = args[3]
            else:
                sample = None
                value_arg = args[2]
            image_format = self.image_resource_format(args[0])
            self.validate_image_store_value_type(texture_type, image_format, value_arg)
            value = self.generate_expression_with_expected(
                value_arg,
                self.image_store_expected_value_type(
                    texture_type, image_format, value_arg
                ),
            )
            value = self.image_store_value_expression(
                texture_type,
                image_format,
                value,
                self.expression_result_type(value_arg),
            )
            if sample is not None:
                return f"imageStore({image_name}, {coord}, {sample}, {value})"
            return f"imageStore({image_name}, {coord}, {value})"

        if is_image_atomic_operation(func_name) and args:
            return self.generate_image_atomic_call(func_name, args)

        texture_type = self.texture_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        if is_texture_compare_operation(func_name):
            return self.generate_texture_compare_call(func_name, args)

        if is_texture_gather_compare_operation(func_name):
            return self.generate_texture_gather_compare_call(func_name, args)

        if is_texture_gather_operation(func_name):
            return self.generate_texture_gather_call(func_name, args)

        if is_texture_query_lod_operation(func_name):
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, coord, _ = parts
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_type(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            coord = self.texture_query_lod_coordinate(texture_type, coord)
            return f"textureQueryLod({texture_name}, {coord})"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            texture_type = self.texture_resource_type(args[0])
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texel_fetch_offset_call(
                    texture_type
                )
            return None

        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            texture_type = self.texture_resource_type(args[0])
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            return None

        if is_projected_texture_operation(func_name) and args:
            texture_type = self.resource_base_type(self.texture_resource_type(args[0]))
            if texture_type == "samplerCube":
                return self.generate_projected_cube_texture_call(func_name, args)
            if texture_type == "samplerCubeArray":
                return self.unsupported_texture_projected_call(
                    func_name,
                    "requires 1D, 2D, 2D-array, or 3D projection coordinates",
                )

        if is_texture_sampling_operation(func_name):
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_call(
                    func_name, texture_type
                )
            if is_texture_sample_offset_operation(
                func_name
            ) and not self.texture_sample_offset_supported(texture_type):
                return self.unsupported_texture_sample_offset_call(
                    func_name, texture_sample_offset_capability_error("GLSL")
                )

        if not is_texture_sampling_operation(
            func_name
        ) or not self.is_explicit_sampler_argument(args):
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None
        texture_name, coord, extra_args = parts
        mapped_args = [texture_name, coord] + [
            self.generate_expression(arg) for arg in extra_args
        ]
        return f"{func_name}({', '.join(mapped_args)})"

    def collect_function_sampler_parameter_indices(self, root):
        sampler_indices = {}
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            name = getattr(value, "name", None)
            params = getattr(value, "parameters", getattr(value, "params", []))
            if name and params:
                indices = []
                for index, param in enumerate(params):
                    param_type = getattr(
                        param, "param_type", getattr(param, "vtype", None)
                    )
                    if self.is_sampler_type(param_type):
                        indices.append(index)
                if indices:
                    sampler_indices[name] = set(indices)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return sampler_indices

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in getattr(root, "global_variables", []) or []:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            base_type = self.resource_base_type(var_type)
            mapped_type = self.map_resource_type_with_format(base_type, node)
            if var_name and (
                self.is_sampler_type(base_type)
                or self.is_opaque_resource_type(mapped_type)
            ):
                resource_names.add(var_name)
        return resource_names

    def validate_global_resource_shadows(self, ast):
        conflicts = collect_non_resource_global_resource_shadows(
            ast,
            self.collect_global_resource_names(ast),
            self.is_inferable_resource_type,
        )
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Non-resource local declaration(s) shadow OpenGL global resource(s): "
                f"{names}"
            )

    def filter_sampler_arguments(self, func_name, args):
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        if not skipped_indices:
            return args
        return [arg for index, arg in enumerate(args) if index not in skipped_indices]

    def collect_resource_array_size_hints(self, ast):
        global_hints, function_hints = collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_sampled_texture_globals(ast),
            function_arrays=self.collect_unsized_sampled_texture_parameters(ast),
            fixed_global_array_sizes=self.collect_fixed_resource_global_sizes(ast),
            fixed_function_array_sizes=self.collect_fixed_resource_parameter_sizes(ast),
            functions=self.collect_functions(ast),
            walk_nodes=self.walk_ast,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=0,
            format_size=lambda size: str(size) if size > 1 else "",
            initial_literal_int_constants=self.initial_literal_int_constants,
        )
        global_hints.update(
            self.collect_structured_buffer_global_array_size_hints(ast, function_hints)
        )
        return global_hints, function_hints

    def collect_unsized_sampled_texture_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_sampled_texture_array_type(vtype):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_structured_buffer_global_array_size_hints(self, ast, function_hints):
        structured_globals = self.collect_unsized_structured_buffer_globals(ast)
        if not structured_globals:
            return {}

        required_sizes = {name: 0 for name in structured_globals}
        fixed_requirements = {}
        functions_by_name = {
            getattr(func, "name", None): func for func in self.collect_functions(ast)
        }
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }

        def raise_fixed_size_conflict(name, left, right):
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{name}': {left} and {right}"
            )

        def record_required_size(name, size):
            size = self.positive_int_value(size)
            if name not in required_sizes or size is None:
                return
            fixed_size = fixed_requirements.get(name)
            if fixed_size is not None and size > fixed_size:
                raise_fixed_size_conflict(name, size, fixed_size)
            required_sizes[name] = max(required_sizes[name], size)

        def record_fixed_requirement(name, size):
            size = self.positive_int_value(size)
            if name not in required_sizes or size is None:
                return
            fixed_size = fixed_requirements.get(name)
            if fixed_size is not None and fixed_size != size:
                raise_fixed_size_conflict(name, fixed_size, size)
            current_size = required_sizes.get(name, 0)
            if current_size and current_size > size:
                raise_fixed_size_conflict(name, current_size, size)
            fixed_requirements[name] = size
            required_sizes[name] = max(current_size, size)

        for node in self.walk_ast(ast):
            if isinstance(node, ArrayAccessNode):
                array_expr = getattr(node, "array", getattr(node, "array_expr", None))
                array_name = self.expression_name(array_expr)
                if array_name not in structured_globals:
                    continue
                index_expr = getattr(node, "index", getattr(node, "index_expr", None))
                index = self.literal_int_value(index_expr, self.literal_int_constants)
                if index is not None and index >= 0:
                    record_required_size(array_name, index + 1)
                continue

            if not isinstance(node, FunctionCallNode):
                continue
            callee_name = self.function_call_name(node)
            callee = functions_by_name.get(callee_name)
            if callee is None:
                continue
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
            callee_hints = function_hints.get(callee_name, {})
            for index, arg in enumerate(args):
                if index >= len(params):
                    break
                arg_name = self.expression_name(arg)
                if arg_name not in structured_globals:
                    continue
                param = params[index]
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                base_type, _ = self.structured_buffer_array_base_type(param_type)
                if base_type is None or not self.is_structured_buffer_type(base_type):
                    continue
                param_name = getattr(param, "name", None)
                fixed_size = self.fixed_resource_array_size(param_type)
                if fixed_size is not None:
                    record_fixed_requirement(arg_name, fixed_size)
                else:
                    record_required_size(arg_name, callee_hints.get(param_name))

        def structured_buffer_param(params, index):
            if index >= len(params):
                return None, None, None
            param = params[index]
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            base_type, _ = self.structured_buffer_array_base_type(param_type)
            if base_type is None or not self.is_structured_buffer_type(base_type):
                return None, None, None
            return param, param_type, getattr(param, "name", None)

        def function_hint_size(func_name, param_name):
            return self.positive_int_value(
                function_hints.get(func_name, {}).get(param_name)
            )

        def record_function_required_size(func_name, param_name, size):
            size = self.positive_int_value(size)
            if not func_name or not param_name or size is None:
                return False
            hints = function_hints.setdefault(func_name, {})
            current_size = self.positive_int_value(hints.get(param_name))
            if current_size is not None and current_size >= size:
                return False
            hints[param_name] = str(size)
            return True

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions_by_name.items():
                caller_hints = function_hints.get(caller_name, {})
                for call in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue
                    params = list(
                        getattr(callee, "parameters", getattr(callee, "params", []))
                    )
                    args = list(
                        getattr(call, "arguments", getattr(call, "args", [])) or []
                    )
                    for index, arg in enumerate(args):
                        _, param_type, param_name = structured_buffer_param(
                            params, index
                        )
                        if (
                            param_type is None
                            or param_name is None
                            or self.fixed_resource_array_size(param_type) is not None
                        ):
                            continue

                        arg_name = self.expression_name(arg)
                        if not arg_name:
                            continue

                        callee_size = function_hint_size(callee_name, param_name)
                        if arg_name in required_sizes:
                            if callee_size is not None:
                                current_size = required_sizes.get(arg_name, 0)
                                record_required_size(arg_name, callee_size)
                                changed = (
                                    required_sizes.get(arg_name, 0) != current_size
                                    or changed
                                )
                            changed = (
                                record_function_required_size(
                                    callee_name,
                                    param_name,
                                    required_sizes.get(arg_name),
                                )
                                or changed
                            )
                            continue

                        caller_size = self.positive_int_value(
                            caller_hints.get(arg_name)
                        )
                        if caller_size is not None:
                            changed = (
                                record_function_required_size(
                                    callee_name, param_name, caller_size
                                )
                                or changed
                            )

        return {
            name: str(size)
            for name, size in required_sizes.items()
            if size is not None and size > 1
        }

    def collect_unsized_structured_buffer_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            base_type, declared_size = self.structured_buffer_array_base_type(vtype)
            if (
                name
                and base_type is not None
                and declared_size is None
                and self.is_structured_buffer_type(base_type)
            ):
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

    def collect_unsized_sampled_texture_parameters(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
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
        for func in self.collect_functions(ast):
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

    def is_unsized_sampled_texture_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_inferable_resource_array_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_inferable_resource_array_type(base_type)

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

    def collect_functions(self, root):
        functions = []
        for node in self.walk_ast(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

    def walk_ast(self, root):
        visited = set()

        def walk(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            yield value

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    yield from walk(child)

        yield from walk(root)

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) == "sampler"

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

    def structured_buffer_array_parameter_info(
        self, vtype, param_name=None, function_name=None
    ):
        base_type, declared_size = self.structured_buffer_array_base_type(vtype)
        if base_type is None or not self.is_structured_buffer_type(base_type):
            return None

        count = declared_size
        if count is None and function_name and param_name:
            hint = self.function_resource_array_size_hints.get(function_name, {}).get(
                param_name
            )
            count = self.positive_int_value(hint)
        if count is None or count <= 0:
            return None
        return {
            "base_type": base_type,
            "element_type": self.structured_buffer_element_type(base_type),
            "count": count,
        }

    def structured_buffer_array_base_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return base_type, self.positive_int_value(vtype.size)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None, None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None, None
        base_type, size = parse_array_type(type_string)
        return base_type, self.positive_int_value(size)

    def collect_unsupported_structured_buffer_array_functions(self, functions):
        unsupported_functions = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            unsupported_params = []
            for index, param in enumerate(
                getattr(func, "parameters", getattr(func, "params", []))
            ):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                base_type, _ = self.structured_buffer_array_base_type(raw_type)
                if base_type is None or not self.is_structured_buffer_type(base_type):
                    continue
                if (
                    self.structured_buffer_array_parameter_info(
                        raw_type, getattr(param, "name", None), func_name
                    )
                    is not None
                ):
                    continue
                unsupported_params.append(
                    {"index": index, "name": getattr(param, "name", None)}
                )
            if unsupported_params:
                unsupported_functions[func_name] = {
                    "indices": {param["index"] for param in unsupported_params},
                    "params": unsupported_params,
                }
        return unsupported_functions

    def skipped_function_parameter_indices(self, func_name):
        skipped = set(self.function_sampler_parameter_indices.get(func_name, set()))
        unsupported = self.unsupported_structured_buffer_array_functions.get(
            func_name, {}
        )
        skipped.update(unsupported.get("indices", set()))
        return skipped

    def unsupported_structured_buffer_array_function_body(
        self, func, unsupported_info, indent
    ):
        indent_str = "    " * indent
        code = ""
        func_name = getattr(func, "name", "<anonymous>")
        for param in unsupported_info.get("params", []):
            param_name = param.get("name") or "<anonymous>"
            code += (
                f"{indent_str}/* unsupported GLSL structured buffer array parameter "
                f"{func_name}.{param_name}: fixed array size is required for helper "
                "lowering */\n"
            )

        return_type = (
            self.current_function_return_type
            or self.type_name_string(getattr(func, "return_type", None))
            or "void"
        )
        mapped_return_type = self.map_type(return_type)
        if mapped_return_type == "void" or self.current_stage_output is not None:
            return f"{code}{indent_str}return;\n"
        return f"{code}{indent_str}return {self.zero_value_expression(return_type)};\n"

    def zero_value_expression(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type == "void":
            return ""
        if mapped_type == "bool":
            return "false"
        if mapped_type == "uint":
            return "0u"
        if mapped_type == "int":
            return "0"
        if mapped_type in {"float", "double"}:
            return "0.0"
        if mapped_type.startswith("bvec"):
            return f"{mapped_type}(false)"
        if mapped_type.startswith("uvec"):
            return f"{mapped_type}(0u)"
        if mapped_type.startswith("ivec"):
            return f"{mapped_type}(0)"
        if mapped_type.startswith(("vec", "dvec", "mat", "dmat")):
            return f"{mapped_type}(0.0)"
        struct = self.structs_by_name.get(mapped_type)
        if struct is not None:
            member_values = []
            for member in getattr(struct, "members", []) or []:
                member_type = getattr(
                    member, "member_type", getattr(member, "vtype", None)
                )
                member_values.append(self.zero_value_expression(member_type))
            return f"{mapped_type}({', '.join(member_values)})"
        return f"{mapped_type}(0)"

    def positive_int_value(self, value):
        literal = self.literal_int_value(value, self.literal_int_constants)
        if literal is None and isinstance(value, int) and not isinstance(value, bool):
            literal = value
        if literal is None and isinstance(value, str) and value.isdigit():
            literal = int(value)
        return literal if literal is not None and literal > 0 else None

    def structured_buffer_block_declaration(
        self, vtype, name, binding, array_size=None
    ):
        element_type = self.structured_buffer_element_type(vtype)
        readonly = (
            "readonly "
            if self.structured_buffer_type_name(vtype) == "StructuredBuffer"
            else ""
        )
        if array_size is not None:
            instance_member = "data"
            self.structured_buffer_instance_members[name] = instance_member
            array_suffix = f"[{array_size}]" if array_size else "[]"
            return (
                f"layout(std430, binding = {binding}) {readonly}buffer "
                f"{name}Buffer {{ {element_type} {instance_member}[]; }} "
                f"{name}{array_suffix};\n"
            )
        return (
            f"layout(std430, binding = {binding}) {readonly}buffer "
            f"{name}Buffer {{ {element_type} {name}[]; }};\n"
        )

    def structured_buffer_counter_resource_name(self, name):
        return f"{name}Counter"

    def structured_buffer_counter_parameter_name(self, name):
        return f"{name}Counter"

    def structured_buffer_counter_block_declaration(
        self, name, binding, array_size=None
    ):
        counter_member = self.structured_buffer_counter_resource_name(name)
        if array_size is not None:
            counter_member = "counter"
            counter_instance = f"{name}Counters"
            self.structured_buffer_counter_members[name] = counter_member
            self.structured_buffer_counter_instances[name] = counter_instance
            array_suffix = f"[{array_size}]" if array_size else "[]"
            return (
                f"layout(std430, binding = {binding}) buffer {name}CounterBuffer "
                f"{{ uint {counter_member}[]; }} {counter_instance}{array_suffix};\n"
            )

        self.structured_buffer_counter_members[name] = counter_member
        return (
            f"layout(std430, binding = {binding}) buffer {name}CounterBuffer "
            f"{{ uint {counter_member}[]; }};\n"
        )

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
            layout_parts = [
                self.attribute_value_to_string(argument) for argument in arguments
            ]
            layout_parts = [part for part in layout_parts if part]
            if layout_parts:
                return ", ".join(layout_parts)
        return "std430"

    def is_shader_record_buffer_block(self, node):
        layout_parts = {
            part.strip().lower().replace("_", "")
            for part in self.glsl_buffer_block_layout(node).split(",")
        }
        return "shaderrecordext" in layout_parts

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

    def glsl_buffer_block_declaration(
        self, node, vtype, name, binding, array_suffix=""
    ):
        block_name = str(self.resource_base_type(vtype))
        struct = self.structs_by_name.get(block_name)
        if struct is None:
            return None

        layout = self.glsl_buffer_block_layout(node)
        memory_qualifiers = self.resource_memory_qualifiers(node)
        qualifier_prefix = f"{memory_qualifiers} " if memory_qualifiers else ""
        layout_prefix = (
            f"layout({layout})"
            if binding is None
            else f"layout({layout}, binding = {binding})"
        )
        code = f"{layout_prefix} {qualifier_prefix}buffer " f"{block_name} {{\n"
        for member in getattr(struct, "members", []) or []:
            code += self.generate_struct_member_declaration(member)
        code += f"}} {name}{array_suffix};\n"
        return code

    def is_sampled_texture_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return (
            mapped_type != "sampler"
            and self.is_opaque_resource_type(mapped_type)
            and not mapped_type.startswith(("image", "iimage", "uimage"))
        )

    def is_storage_image_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return self.is_opaque_resource_type(mapped_type) and is_glsl_storage_image_type(
            mapped_type
        )

    def is_inferable_resource_array_type(self, vtype):
        return self.is_sampled_texture_type(vtype) or self.is_storage_image_type(vtype)

    def is_resource_array_hint_type(self, vtype):
        return self.is_inferable_resource_array_type(
            vtype
        ) or self.is_structured_buffer_type(vtype)

    def is_inferable_resource_type(self, vtype):
        return (
            self.is_sampler_type(vtype)
            or self.is_structured_buffer_type(vtype)
            or self.is_inferable_resource_array_type(vtype)
        )

    def resource_base_type(self, vtype):
        if vtype is None:
            return ""
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return self.resource_base_type(vtype.element_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype = self.convert_type_node_to_string(vtype)
        vtype = str(vtype)
        if "[" in vtype and "]" in vtype:
            base_type, _ = parse_array_type(vtype)
            return base_type
        return vtype

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def resource_node_type(self, node):
        return getattr(
            node,
            "var_type",
            getattr(node, "param_type", getattr(node, "vtype", "float")),
        )

    def resource_node_name(self, node, default=None):
        return getattr(node, "name", getattr(node, "variable_name", default))

    def resource_declaration_shape(self, node):
        node_type = self.resource_node_type(node)
        node_name = self.resource_node_name(node, "")
        resource_count = 1
        array_size = None
        array_suffix = ""

        if (
            hasattr(node_type, "element_type")
            and str(type(node_type)).find("ArrayType") != -1
        ):
            vtype = self.convert_type_node_to_string(node_type.element_type)
            array_size = (
                self.generate_expression(node_type.size)
                if node_type.size
                else (
                    self.resource_array_size_hints.get(node_name, "")
                    if self.is_resource_array_hint_type(vtype)
                    else ""
                )
            )
            array_suffix = f"[{array_size}]" if array_size else "[]"
            resource_count = self.resource_array_count(
                node_type.size if node_type.size else array_size
            )
            return vtype, array_size, array_suffix, resource_count

        if hasattr(node_type, "name") or hasattr(node_type, "element_type"):
            return self.convert_type_node_to_string(node_type), None, "", 1

        vtype = str(node_type) if node_type is not None else "float"
        if "[" in vtype and "]" in vtype:
            base_type, parsed_size = parse_array_type(vtype)
            array_size = parsed_size
            array_suffix = f"[{parsed_size}]" if parsed_size else "[]"
            resource_count = self.resource_array_count(parsed_size)
            return base_type, array_size, array_suffix, resource_count

        return vtype, None, "", resource_count

    def resource_declaration_identity(self, node):
        vtype, _, array_suffix, resource_count = self.resource_declaration_shape(node)
        if self.is_glsl_buffer_block_variable(node, vtype):
            return (
                "buffer binding",
                "glsl_buffer_block",
                str(self.resource_base_type(vtype)),
                array_suffix,
                resource_count,
                self.glsl_buffer_block_layout(node),
                self.resource_memory_qualifiers(node),
            )
        if self.is_structured_buffer_type(vtype):
            return (
                "buffer binding",
                self.structured_buffer_type_name(vtype),
                self.structured_buffer_element_type(vtype),
                array_suffix,
                resource_count,
            )

        mapped_type = self.map_resource_type_with_format(vtype, node)
        if mapped_type == "sampler" or not self.is_opaque_resource_type(mapped_type):
            return None
        namespace = (
            "image binding" if self.is_storage_image_type(vtype) else "texture binding"
        )
        return (
            namespace,
            mapped_type,
            array_suffix,
            resource_count,
            self.image_format_qualifier(mapped_type, node),
            self.resource_memory_qualifiers(node),
        )

    def global_resource_shape(self, node):
        vtype, _, _, resource_count = self.resource_declaration_shape(node)
        return vtype, resource_count

    def global_resource_binding_metadata(self, node):
        var_name = self.resource_node_name(node)
        if not var_name:
            return None

        vtype, resource_count = self.global_resource_shape(node)
        if self.is_glsl_buffer_block_variable(node, vtype):
            namespace = "buffer binding"
        elif self.is_structured_buffer_type(vtype):
            namespace = "buffer binding"
        else:
            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type == "sampler":
                return None
            if not self.is_opaque_resource_type(mapped_type):
                return None
            namespace = (
                "image binding"
                if self.is_storage_image_type(vtype)
                else "texture binding"
            )

        binding = self.explicit_resource_binding_index(node)
        if binding is None:
            return None
        return namespace, binding, resource_count, var_name

    def reserve_explicit_global_resource_bindings(self, global_vars, used_bindings):
        for node in global_vars:
            metadata = self.global_resource_binding_metadata(node)
            if metadata is None:
                continue
            namespace, binding, resource_count, var_name = metadata
            self.reserve_resource_binding_range(
                used_bindings,
                "OpenGL",
                namespace,
                binding,
                resource_count,
                var_name,
            )

    def next_resource_binding(self, binding_cursors, namespace):
        return binding_cursors.get(namespace, 0)

    def next_available_resource_binding(
        self, used_bindings, binding_cursors, namespace, count
    ):
        count = max(count or 1, 1)
        binding = self.next_resource_binding(binding_cursors, namespace)
        ranges = used_bindings.get(namespace, [])
        while True:
            end = binding + count - 1
            conflict_end = None
            for used_start, used_end, _ in ranges:
                if binding <= used_end and used_start <= end:
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
            if conflict_end is None:
                return binding
            binding = conflict_end + 1

    def advance_resource_binding(self, binding_cursors, namespace, start, count):
        count = max(count or 1, 1)
        binding_cursors[namespace] = max(
            binding_cursors.get(namespace, 0), start + count
        )

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
            return f"{namespace} {start}"
        return f"{namespace} {start}-{end}"

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def initial_literal_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)
        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def visible_literal_int_constants(self, func):
        visible_constants = self.initial_literal_int_constants(func)

        for node in self.walk_ast(getattr(func, "body", [])):
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

    def map_type(self, vtype):
        """Map types to GLSL equivalents, handling both strings and TypeNode objects."""
        if vtype is None:
            return "float"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if self.is_ray_query_type_name(vtype_str):
            return "rayQueryEXT"

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_type(base_type)
            return f"{base_mapped}{array_suffix}"

        generic_enum_type = generic_enum_specialized_type_name(self, vtype_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_struct_type = generic_struct_specialized_type_name(self, vtype_str)
        if generic_struct_type is not None:
            return generic_struct_type

        if vtype_str in getattr(self, "enum_type_names", set()):
            return "int"

        if vtype_str in getattr(self, "enum_struct_type_names", set()):
            return vtype_str

        return self.type_mapping.get(vtype_str, vtype_str)

    def is_ray_query_type(self, vtype):
        if vtype is None:
            return False
        return self.is_ray_query_type_name(self.type_name_string(vtype))

    def is_acceleration_structure_type(self, vtype):
        if vtype is None:
            return False
        return self.resource_base_type(vtype) == "accelerationStructureEXT"

    def is_ray_query_type_name(self, type_name):
        type_name = str(type_name)
        return (
            type_name == "rayQueryEXT"
            or type_name == "RayQuery"
            or type_name.startswith("RayQuery<")
        )

    def map_resource_parameter_type_with_hint(
        self, vtype, node=None, function_name=None
    ):
        if vtype is None:
            return self.map_type(vtype)
        if self.is_structured_buffer_type(vtype):
            return f"{self.structured_buffer_element_type(vtype)}[]"

        function_hints = self.function_resource_array_size_hints.get(function_name, {})
        param_name = getattr(node, "name", None)

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if self.is_inferable_resource_array_type(base_type):
                array_size = (
                    self.expression_to_string(vtype.size)
                    if vtype.size is not None
                    else function_hints.get(param_name, "")
                )
                mapped_type = self.map_image_base_type_with_format(base_type, node)
                return (
                    f"{mapped_type}[{array_size}]" if array_size else f"{mapped_type}[]"
                )

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                base_type, array_suffix = split_array_type_suffix(type_string)
                if self.is_inferable_resource_array_type(base_type):
                    mapped_type = self.map_image_base_type_with_format(base_type, node)
                    if array_suffix == "[]":
                        array_size = function_hints.get(param_name, "")
                        return (
                            f"{mapped_type}[{array_size}]"
                            if array_size
                            else f"{mapped_type}[]"
                        )
                    return f"{mapped_type}{array_suffix}"

        return self.map_resource_type_with_format(vtype, node)

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
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
        if explicit_format in {
            "r8",
            "r8_snorm",
            "r16",
            "r16_snorm",
            "r16f",
            "r32f",
            "r8i",
            "r16i",
            "r32i",
            "r8ui",
            "r16ui",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg8i",
            "rg16i",
            "rg8ui",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba32f",
            "rgba8i",
            "rgba16i",
            "rgba32i",
            "rgba8ui",
            "rgba16ui",
            "rgba32ui",
        }:
            if explicit_format in {
                "r8",
                "r8_snorm",
                "r16",
                "r16_snorm",
                "r16f",
                "rg8",
                "rg8_snorm",
                "rg16",
                "rg16_snorm",
                "rg16f",
                "rg32f",
                "rgba8",
                "rgba8_snorm",
                "rgba16",
                "rgba16_snorm",
                "rgba16f",
                "rgba32f",
            }:
                format_class = "r32f"
            elif explicit_format.endswith("ui"):
                format_class = "r32ui"
            elif explicit_format.endswith("i"):
                format_class = "r32i"
            else:
                format_class = explicit_format
            format_type_map = {
                "image1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "iimage1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "uimage1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "image2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "iimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "uimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "image3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "iimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "uimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "image1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "iimage1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "uimage1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "image2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "iimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "uimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "image2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "iimage2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "uimage2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "image2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "iimage2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "uimage2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "imageCube": {"r32f": "imageCube"},
            }
            mapped_type = format_type_map.get(base_type, {}).get(format_class)
            if mapped_type:
                return mapped_type

        return self.map_type(vtype)

    def is_opaque_resource_type(self, vtype):
        return vtype in {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler3D",
            "samplerCube",
            "sampler2DArray",
            "samplerCubeArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "sampler2DRect",
            "samplerBuffer",
            "sampler2DMS",
            "sampler2DMSArray",
            "isampler2D",
            "usampler2D",
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
            "image2DMS",
            "image2DMSArray",
            "imageBuffer",
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
            "atomic_uint",
            "accelerationStructureEXT",
        }

    def supported_image_formats(self):
        return supported_image_formats()

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

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "volatile",
            "restrict",
            "readonly",
            "writeonly",
            "readwrite",
        }

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
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

    def explicit_resource_binding_index(self, node):
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in {"binding", "buffer", "sampler", "texture"}:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], ("b", "s", "t", "u"))
            else:
                binding = None
            if binding is not None:
                return binding
        return None

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
                or (
                    getattr(attr, "name", None)
                    and str(getattr(attr, "name")).lower() == "glsl_buffer_block"
                )
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def glsl_stage_control_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("glsl_"):
            normalized = normalized[len("glsl_") :]

        valid_names = {
            "cw",
            "ccw",
            "domain",
            "early_fragment_tests",
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
            "max_vertices",
            "max_primitives",
            "maxprimitivecount",
            "maxvertexcount",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
            "point_mode",
            "points",
            "quads",
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
            if self.glsl_stage_control_attribute_name(attr):
                continue
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or (
                    getattr(attr, "name", None)
                    and str(getattr(attr, "name")).lower() == "glsl_buffer_block"
                )
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def image_format_qualifier(self, vtype, node=None):
        explicit_format = explicit_image_format(node, self.attribute_value_to_string)
        if explicit_format:
            return explicit_format
        if vtype in {
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
        }:
            return "rgba32f"
        if vtype in {
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
        }:
            return "r32i"
        if vtype in {
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
        }:
            return "r32ui"
        return None

    def opaque_resource_layout(self, vtype, binding, node=None):
        image_format = self.image_format_qualifier(vtype, node)
        if image_format:
            return f"layout({image_format}, binding = {binding})"
        return f"layout(binding = {binding})"

    def resource_memory_qualifiers(self, node):
        supported = {
            "coherent",
            "globallycoherent",
            "volatile",
            "restrict",
            "readonly",
            "writeonly",
        }
        qualifiers = set()

        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier = str(qualifier).lower()
            if qualifier in supported:
                qualifiers.add(qualifier)

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported:
                qualifiers.add(attr_name)

        if "globallycoherent" in qualifiers:
            qualifiers.add("coherent")

        order = ("coherent", "volatile", "restrict", "readonly", "writeonly")
        return " ".join(qualifier for qualifier in order if qualifier in qualifiers)

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
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
            "MOD": "%",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "LOGICAL_AND": "&&",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to the corresponding GLSL builtin or layout."""
        if semantic is not None:
            return f"{self.semantic_map.get(semantic, semantic)}"
        else:
            return ""

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        generic_args = getattr(type_node, "generic_args", [])
        if hasattr(type_node, "name") and generic_args:
            args = ", ".join(
                self.convert_type_node_to_string(arg) for arg in generic_args
            )
            return f"{type_node.name}<{args}>"
        if hasattr(type_node, "name"):
            return type_node.name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.rows == type_node.cols:
                return f"mat{type_node.rows}"
            return f"mat{type_node.cols}x{type_node.rows}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return f"{element_type}[{type_node.size}]"
                    else:
                        size_str = self.expression_to_string(type_node.size)
                        return f"{element_type}[{size_str}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"vec{size}"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
                elif element_type == "bool":
                    return f"bvec{size}"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        else:
            return self.generate_expression(expr)

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        semantic_attrs = [
            "position",
            "color",
            "texcoord",
            "normal",
            "tangent",
            "binormal",
            "POSITION",
            "COLOR",
            "TEXCOORD",
            "NORMAL",
            "TANGENT",
            "BINORMAL",
            "TEXCOORD0",
            "TEXCOORD1",
            "TEXCOORD2",
            "TEXCOORD3",
        ]

        for attr in attributes:
            if hasattr(attr, "name") and attr.name in semantic_attrs:
                return attr.name
        return None

    def generate_array_declaration(self, stmt, indent):
        indent_str = "    " * indent
        element_type = self.map_type(stmt.element_type)
        size = get_array_size_from_node(stmt)

        if size is None:
            # In GLSL, dynamic sized arrays need special handling
            # For instance in shader storage blocks, but for simple cases:
            return f"{indent_str}{element_type} {stmt.name}[];\n"
        else:
            return f"{indent_str}{element_type} {stmt.name}[{size}];\n"

    def generate_struct_member_declaration(self, member, indent="    "):
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            if member.size:
                return f"{indent}{self.map_type(element_type)} {member.name}[{member.size}];\n"
            return f"{indent}{self.map_type(element_type)} {member.name}[];\n"

        if hasattr(member, "member_type"):
            if str(type(member.member_type)).find("ArrayType") != -1:
                member_type_str = self.convert_type_node_to_string(member.member_type)
                member_type = self.map_type(member_type_str)
                declaration = format_c_style_array_declaration(member_type, member.name)
                return f"{indent}{declaration};\n"

            member_type_str = self.convert_type_node_to_string(member.member_type)
            member_type = self.map_type(member_type_str)
            return f"{indent}{member_type} {member.name};\n"

        if hasattr(member, "vtype"):
            member_type = self.map_type(member.vtype)
            return f"{indent}{member_type} {member.name};\n"

        return f"{indent}float {member.name};\n"

    def generate_struct(self, node):
        code = f"struct {node.name} {{\n"
        for member in getattr(node, "members", []) or []:
            code += self.generate_struct_member_declaration(member)
        code += "};\n"
        return code

    def generate_expression_statement(self, stmt):
        """Generate code for expression statements."""
        if hasattr(stmt, "expression"):
            expr = self.generate_expression(stmt.expression)
            return expr
        else:
            # Fallback for direct expression
            return self.generate_expression(stmt)
