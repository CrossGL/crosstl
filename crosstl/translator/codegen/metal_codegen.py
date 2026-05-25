"""CrossGL-to-Metal code generator."""

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
    format_array_type,
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
from .glsl_buffer_layout import (
    byte_offset_expression,
    collect_lowered_glsl_buffer_blocks,
    glsl_buffer_compound_binary_operator,
    glsl_buffer_block_node_type,
    matrix_column_offsets,
    vector_component_offsets,
)
from .image_access_contracts import (
    TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_INTRINSIC_NAMES,
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    default_storage_image_channel_count,
    explicit_image_access,
    explicit_image_format,
    floating_coordinate_dimension_from_type_name,
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
    image_format_channel_count,
    image_format_component_kind,
    image_format_result_type,
    image_format_component_type,
    image_format_vector_type,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_atomic_helper_descriptor_fields,
    image_atomic_helper_resource_metadata,
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
    is_floating_scalar_type_name,
    is_integer_coordinate_type_name,
    is_integer_scalar_type_name,
    is_numeric_scalar_type_name,
    is_metal_float_image_resource,
    is_metal_integer_image_type,
    is_metal_storage_image_resource,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_basic_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_grad_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_lod_operation,
    is_projected_texture_operation,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_resource_access_attribute,
    is_scalar_image_format,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_any_offset_operation,
    is_texture_compare_operation,
    is_texture_compare_basic_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_offset_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_compare_offset_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_offset_operation,
    is_texture_gather_single_offset_operation,
    is_texture_query_lod_operation,
    is_texture_query_levels_operation,
    is_texture_sample_basic_offset_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_offset_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_lod_offset_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_operation,
    is_texture_sampling_operation,
    is_texture_sampling_offset_operation,
    is_texture_sample_offset_operation,
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
    image_resource_metadata,
    record_explicit_image_metadata,
    projected_texture_offset_capability_error,
    projected_texture_extra_argument_count_error,
    resource_query_method_size_descriptor,
    resolve_image_atomic_component_kind,
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
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
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
    texture_sample_offset_extra_argument_count_error,
    texture_sample_offset_capability_error,
    texture_multisample_sample_type_error,
    validate_texture_operation_arity,
    requires_integer_coordinate,
    unsupported_image_atomic_expression,
    unsupported_cube_texel_fetch_expression,
    unsupported_multisample_image_atomic_expression,
    unsupported_multisample_image_store_expression,
    unsupported_multisample_texture_call_vector_expression,
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_multisample_texel_fetch_offset_expression,
    unsupported_projected_texture_call_expression,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texture_gather_compare_call_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_compare_operation_error,
    unsupported_texture_offset_call_expression,
    unsupported_texture_offset_operation_error,
    unsupported_projected_texture_operation_error,
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

    def __init__(self):
        """Initialize Metal type maps and per-generation resource state."""
        self.current_shader = None
        self.vertex_item = None
        self.fragment_item = None
        self.gl_position = False
        self.char_mapper = CharTypeMapper()
        self.texture_variables = []
        self.sampler_variables = []
        self.structured_buffer_variables = []
        self.cbuffer_variables = []
        self.cbuffer_binding_indices = {}
        self.cbuffer_parameter_names = {}
        self.cbuffer_member_references = {}
        self.ambiguous_cbuffer_members = set()
        self.cbuffers_by_name = {}
        self.user_function_names = set()
        self.function_parameter_names = {}
        self.function_return_types = {}
        self.function_image_access_requirements = {}
        self.function_cbuffer_dependencies = {}
        self.function_global_resource_dependencies = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function_return_wrapper = None
        self.current_expression_expected_type = None
        self.current_generic_function_substitutions = {}
        self.local_variable_types = {}
        self.struct_member_types = {}
        self.structs_by_name = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.struct_constructor_uses_braces = True
        self.glsl_buffer_block_struct_names = set()
        self.lowered_glsl_buffer_blocks = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_glsl_buffer_block_parameters = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.glsl_buffer_block_variables = []
        self.metal_temp_variable_index = 0
        self.generated_return_wrapper_struct_names = set()
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
            "float": "float",
            "half": "half",
            "float16": "half",
            "min16float": "half",
            "min10float": "half",
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
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_IsFrontFace": "is_front_facing",
            "gl_PrimitiveID": "primitive_id",
            "POSITION": "attribute(0)",
            "NORMAL": "attribute(1)",
            "TANGENT": "attribute(2)",
            "BINORMAL": "attribute(3)",
            "TEXCOORD": "attribute(4)",
            "TEXCOORD0": "attribute(5)",
            "TEXCOORD1": "attribute(6)",
            "TEXCOORD2": "attribute(7)",
            "TEXCOORD3": "attribute(8)",
            "TEXCOORD4": "attribute(9)",
            "TEXCOORD5": "attribute(10)",
            "TEXCOORD6": "attribute(11)",
            "TEXCOORD7": "attribute(12)",
            # Vertex outputs
            "gl_Position": "position",
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
            # Additional Metal-specific attributes
            "gl_FragCoord": "position",
            "gl_FrontFacing": "is_front_facing",
            "gl_PointCoord": "point_coord",
            # Compute shader specific
            "gl_GlobalInvocationID": "thread_position_in_grid",
            "gl_LocalInvocationID": "thread_position_in_threadgroup",
            "gl_WorkGroupID": "threadgroup_position_in_grid",
            "gl_LocalInvocationIndex": "thread_index_in_threadgroup",
            "gl_WorkGroupSize": "threads_per_threadgroup",
            "gl_NumWorkGroups": "threadgroups_per_grid",
            # Ray tracing / payload semantics
            "payload": "payload",
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
        self.sampler_variables = []
        self.structured_buffer_variables = []
        self.glsl_buffer_block_variables = []
        self.lowered_glsl_buffer_blocks = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.metal_temp_variable_index = 0
        self.generated_return_wrapper_struct_names = set()
        self.cbuffer_variables = getattr(ast, "cbuffers", []) or []
        self.cbuffer_binding_indices = {}
        self.cbuffers_by_name = {
            cbuffer.name: cbuffer
            for cbuffer in self.cbuffer_variables
            if getattr(cbuffer, "name", None)
        }
        all_functions = self.all_functions(ast)
        self.user_function_names = {
            func.name for func in all_functions if getattr(func, "name", None)
        }
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
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.function_global_resource_dependencies = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.required_image_atomic_compare_helpers = set()
        self.current_glsl_buffer_block_parameters = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
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
        global_vars = deduplicate_named_declarations(
            list(getattr(ast, "global_variables", []) or [])
            + collect_stage_local_variables(
                ast, target_stage, self.is_stage_local_resource_variable
            ),
            "Metal resource",
        )
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(
                list(global_vars) + self.collect_function_parameters(all_functions)
            )
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.validate_global_resource_shadows(ast)
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function_return_wrapper = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
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
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        pre_lines = []
        for directive in preprocessors:
            if isinstance(directive, PreprocessorNode):
                line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line:
                pre_lines.append(line)
        if pre_lines:
            code += "\n".join(pre_lines) + "\n"
        if not any("metal_stdlib" in line for line in pre_lines):
            code += "#include <metal_stdlib>\n"
        code += "using namespace metal;\n"
        code += "\n"
        code += generate_enum_constants(
            self, self.plain_enums + self.struct_payload_enums
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
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

        for node in structs:
            if isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                if node.name in self.generic_struct_definitions:
                    continue
                if node.name in self.lowered_glsl_buffer_block_struct_names:
                    continue
                if node.name in self.glsl_buffer_block_struct_names:
                    code += self.glsl_buffer_block_diagnostic(
                        "Metal", node.name, None, None
                    )
                    code += self.unsupported_glsl_buffer_block_struct_placeholder(
                        "Metal", node.name
                    )
                    continue
                self.validate_struct_member_semantic_types(node)
                code += f"struct {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        # Handle array types in structs
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in Metal use array<type>
                            code += f"    array<{self.map_type(element_type)}> {member.name};\n"
                    else:
                        semantic = self.semantic_from_node(member)
                        if hasattr(member, "member_type"):
                            if str(type(member.member_type)).find("ArrayType") != -1:
                                member_type_str = self.convert_type_node_to_string(
                                    member.member_type
                                )
                                member_type = self.map_type(member_type_str)
                                declaration = format_c_style_array_declaration(
                                    member_type, member.name
                                )
                                semantic_attr = (
                                    self.map_semantic(semantic) if semantic else ""
                                )
                                if member.member_type.size is None:
                                    base_type, _ = split_array_type_suffix(member_type)
                                    code += (
                                        f"    array<{base_type}> {member.name}"
                                        f"{semantic_attr};\n"
                                    )
                                else:
                                    code += f"    {declaration}{semantic_attr};\n"
                                continue  # Skip the normal member_type handling
                            else:
                                member_type_str = self.convert_type_node_to_string(
                                    member.member_type
                                )
                                member_type = self.map_type(member_type_str)
                        elif hasattr(member, "vtype"):
                            member_type = self.map_type(member.vtype)
                        else:
                            member_type = "float"

                        semantic_attr = self.map_semantic(semantic) if semantic else ""
                        code += f"    {member_type} {member.name}{semantic_attr};\n"
                code += "};\n"

        texture_register = 0
        sampler_register = 0
        buffer_register = 0
        used_resource_bindings = {}
        buffer_register = self.reserve_cbuffer_bindings(used_resource_bindings)
        self.reserve_explicit_global_resource_bindings(
            global_vars, used_resource_bindings
        )
        for i, node in enumerate(global_vars):
            # Handle both old and new AST variable structures
            resource_count = 1
            array_size = None
            if hasattr(node, "var_type"):
                if hasattr(node.var_type, "name") or hasattr(
                    node.var_type, "element_type"
                ):
                    # Check if it's an ArrayType and handle specially for global variables
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
                        # Use the proper type conversion for TypeNode objects
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
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "Metal",
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
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
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "Metal",
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                )
                self.structured_buffer_variables.append(
                    (node, binding, vtype, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
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
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "Metal",
                    "texture",
                    binding,
                    resource_count,
                    var_name,
                )
                self.texture_variables.append((node, binding, mapped_type, array_size))
                self.texture_variable_types[node.name] = mapped_type
                record_explicit_image_metadata(
                    node.name,
                    node,
                    self.attribute_value_to_string,
                    image_formats=self.image_variable_formats,
                )
                texture_register = max(texture_register, binding + resource_count)
            elif vtype in ["sampler"]:
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
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "Metal",
                    "sampler",
                    binding,
                    resource_count,
                    var_name,
                )
                self.sampler_variables.append((node, binding, array_size))
                sampler_register = max(sampler_register, binding + resource_count)
            else:
                code += f"{self.map_type(vtype)} {node.name}{array_suffix};\n"

        self.function_global_resource_dependencies = (
            self.collect_function_global_resource_dependencies(all_functions)
        )

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        stage_entry_names = self.stage_entry_names(ast, target_stage)

        functions = getattr(ast, "functions", [])
        functions_code = ""
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)
            qualifier_name = normalize_stage_name(qualifier)

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
                )
            else:
                functions_code += self.generate_function(func)

        # Handle shader stages (new AST structure)
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
                        entry_name=stage_entry_names.get(id(stage.entry_point)),
                    )

        code += self.generate_image_atomic_compare_helpers()
        code += functions_code
        return code

    def generate_constants(self, ast):
        code = ""
        for node in getattr(ast, "constants", []) or []:
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = self.generate_constant_expression(value)
            code += f"constant {self.map_type(const_type)} {name} = {value_code};\n"

        return f"{code}\n" if code else ""

    def generate_constant_expression(self, expr):
        value_code = self.generate_expression(expr)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
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
                            # Dynamic arrays in buffer blocks
                            code += f"    array<{self.map_type(element_type)}> {member.name};\n"
                    else:
                        # Handle both old and new AST member structures
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
                            # Dynamic arrays in buffer blocks
                            code += f"    array<{self.map_type(element_type)}> {member.name};\n"
                    else:
                        # Handle both old and new AST member structures
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

    def generate_function(self, func, indent=0, shader_type=None, entry_name=None):
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
        texture_parameters = {}
        image_format_parameters = {}
        previous_function_name = self.current_function_name
        previous_function_return_type = self.current_function_return_type
        previous_function_return_wrapper = self.current_function_return_wrapper
        previous_local_variable_types = self.local_variable_types
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
        self.current_function_name = getattr(func, "name", None)
        self.current_function_return_wrapper = None
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        (
            self.current_glsl_buffer_block_parameters,
            self.current_glsl_buffer_block_parameter_failures,
            self.current_glsl_buffer_block_parameter_struct_failures,
        ) = self.collect_lowered_glsl_buffer_block_parameters(param_list)
        if shader_type in {"vertex", "fragment", "compute"}:
            self.validate_stage_parameter_resource_bindings(
                param_list, self.current_function_name
            )
        if shader_type in {"vertex", "fragment"}:
            self.validate_graphics_builtin_parameter_types(param_list, shader_type)
        if shader_type == "compute":
            self.validate_compute_builtin_parameter_types(param_list)
        self.current_unsupported_glsl_buffer_block_parameters = (
            self.collect_unsupported_glsl_buffer_block_parameter_names(param_list)
        )
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.local_variable_types = {}
        for p in param_list:
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

            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
            elif self.is_texture_or_image_resource_type(raw_param_type):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=image_format_parameters,
                )
            param_type = self.map_resource_type_with_format(raw_param_type, p)

            semantic = self.semantic_from_node(p)

            param_attr = self.parameter_attribute(
                raw_param_type, semantic, shader_type, p
            )
            declaration = self.format_parameter_declaration(
                raw_param_type, param_type, p.name, p
            )
            params.append(f"{declaration}{param_attr}")

        if shader_type == "compute":
            existing_param_names = {getattr(p, "name", None) for p in param_list}
            for name, param_type, attribute in self.required_compute_builtin_parameters(
                func
            ):
                if name not in existing_param_names:
                    params.append(f"{param_type} {name} [[{attribute}]]")
                    reserved_parameter_names.add(name)

        reserved_parameter_names.update(self.global_resource_parameter_names())
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            self.cbuffer_variables, reserved_names=reserved_parameter_names
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            self.cbuffer_variables
        )

        params_str = ", ".join(params)
        if shader_type is None:
            params_str = self.append_required_cbuffer_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_global_resource_parameters(
                params_str, self.current_function_name
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
        self.current_function_return_type = raw_return_type
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
            self.local_variable_types = previous_local_variable_types
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
            self.current_glsl_buffer_block_parameter_failures = (
                previous_glsl_buffer_block_parameter_failures
            )
            self.current_glsl_buffer_block_parameter_struct_failures = (
                previous_glsl_buffer_block_parameter_struct_failures
            )
            return code

        if shader_type == "vertex":
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name
            )
            function_name = entry_name or f"vertex_{func.name}"
            return_wrapper = self.function_return_semantic_wrapper(
                func, raw_return_type, return_type, shader_type, function_name
            )
            if return_wrapper is not None:
                code += self.generate_return_wrapper_struct(return_wrapper)
                return_type = return_wrapper["struct_name"]
            self.current_function_return_wrapper = return_wrapper
            code += f"vertex {return_type} {function_name}({params_str}) {{\n"
        elif shader_type == "fragment":
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name
            )
            function_name = entry_name or f"fragment_{func.name}"
            return_wrapper = self.function_return_semantic_wrapper(
                func, raw_return_type, return_type, shader_type, function_name
            )
            if return_wrapper is not None:
                code += self.generate_return_wrapper_struct(return_wrapper)
                return_type = return_wrapper["struct_name"]
            self.current_function_return_wrapper = return_wrapper
            code += f"fragment {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["compute", "ray_generation"]:
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name
            )
            function_name = entry_name or f"kernel_{func.name}"
            code += f"kernel {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["mesh", "object", "task", "amplification"]:
            stage_keyword = "mesh" if shader_type == "mesh" else "object"
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            code += f"{stage_keyword} {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in [
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
        ]:
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
            stage_keyword = rt_stage_map.get(shader_type, shader_type)
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            code += f"{stage_keyword} {return_type} {function_name}({params_str}) {{\n"
        else:
            semantic = self.semantic_from_node(func)
            function_name = entry_name or func.name
            code += f"{return_type} {function_name}({params_str}) {self.map_semantic(semantic)} {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_image_format_parameters = self.current_image_format_parameters
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = texture_parameters
        self.current_image_format_parameters = image_format_parameters
        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, 1)
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_texture_parameters = previous_texture_parameters
        self.current_image_format_parameters = previous_image_format_parameters
        self.current_function_name = previous_function_name
        self.current_function_return_type = previous_function_return_type
        self.current_function_return_wrapper = previous_function_return_wrapper
        self.local_variable_types = previous_local_variable_types
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
        self.current_glsl_buffer_block_parameter_failures = (
            previous_glsl_buffer_block_parameter_failures
        )
        self.current_glsl_buffer_block_parameter_struct_failures = (
            previous_glsl_buffer_block_parameter_struct_failures
        )

        code += "}\n\n"
        return code

    def required_compute_builtin_parameters(self, func):
        used_names = self.used_compute_builtin_names(getattr(func, "body", []))
        builtin_parameters = [
            ("gl_GlobalInvocationID", "uint3", "thread_position_in_grid"),
            ("gl_LocalInvocationID", "uint3", "thread_position_in_threadgroup"),
            ("gl_WorkGroupID", "uint3", "threadgroup_position_in_grid"),
            ("gl_LocalInvocationIndex", "uint", "thread_index_in_threadgroup"),
            ("gl_WorkGroupSize", "uint3", "threads_per_threadgroup"),
            ("gl_NumWorkGroups", "uint3", "threadgroups_per_grid"),
        ]
        return [
            parameter for parameter in builtin_parameters if parameter[0] in used_names
        ]

    def used_compute_builtin_names(self, body):
        builtin_names = {
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        used_names = set()
        for node in self.iter_ast_nodes(body):
            if hasattr(node, "__class__") and "Identifier" in str(node.__class__):
                name = getattr(node, "name", "")
                base_name = name.split(".", 1)[0]
                if base_name in builtin_names:
                    used_names.add(base_name)
        return used_names

    def validate_compute_builtin_parameter_types(self, parameters):
        expected_types = {
            "thread_position_in_grid": "uint3",
            "thread_position_in_threadgroup": "uint3",
            "threadgroup_position_in_grid": "uint3",
            "thread_index_in_threadgroup": "uint",
            "threads_per_threadgroup": "uint3",
            "threadgroups_per_grid": "uint3",
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = expected_types.get(metal_semantic)
            if expected_type is None:
                continue
            actual_type = self.map_type(self.parameter_raw_type(parameter))
            if actual_type != expected_type:
                name = getattr(parameter, "name", "<anonymous>")
                raise ValueError(
                    f"Metal compute semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires parameter '{name}' to "
                    f"have type {expected_type}, got {actual_type}"
                )

    def validate_graphics_builtin_parameter_types(self, parameters, stage_name):
        expected_types = {
            "vertex_id": "uint",
            "instance_id": "uint",
            "primitive_id": "uint",
            "is_front_facing": "bool",
            "position": "float4",
            "point_coord": "float2",
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = expected_types.get(metal_semantic)
            if expected_type is None:
                continue
            actual_type = self.map_type(self.parameter_raw_type(parameter))
            if actual_type != expected_type:
                name = getattr(parameter, "name", "<anonymous>")
                raise ValueError(
                    f"Metal {stage_name} semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires parameter '{name}' to "
                    f"have type {expected_type}, got {actual_type}"
                )

    def validate_struct_member_semantic_types(self, struct_node):
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = self.struct_member_builtin_semantic_type(metal_semantic)
            if expected_type is None:
                continue
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
        }
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
        is_fragment_output = semantic_text == "gl_FragDepth" or bool(
            metal_semantic and metal_semantic.startswith("color(")
        )
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
        output_types = {
            "gl_Position": "float4",
            "gl_FragDepth": "float",
        }
        if semantic in output_types:
            return output_types[semantic]
        if metal_semantic and metal_semantic.startswith("color("):
            return "float4"
        invalid_return_semantics = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_IsFrontFace",
            "gl_PointSize",
            "gl_FragCoord",
            "gl_FrontFacing",
            "gl_PointCoord",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        if semantic in invalid_return_semantics:
            return "invalid"
        return None

    def canonical_metal_semantic(self, semantic):
        if semantic is None:
            return None
        mapped_semantic = self.semantic_map.get(str(semantic), str(semantic))
        if mapped_semantic.startswith("[[") and mapped_semantic.endswith("]]"):
            return mapped_semantic[2:-2]
        return mapped_semantic

    def append_global_resource_parameters(self, params_str, func_name=None):
        resource_params = []
        dependencies = (
            self.function_global_resource_dependencies.get(func_name, set())
            if func_name
            else None
        )
        if self.cbuffer_variables:
            for cbuffer in self.cbuffer_variables:
                binding = self.cbuffer_binding_indices.get(id(cbuffer), 0)
                parameter_name = self.cbuffer_parameter_name(cbuffer)
                resource_params.append(
                    f"constant {cbuffer.name}& {parameter_name} [[buffer({binding})]]"
                )
        if self.texture_variables:
            for (
                texture_variable,
                i,
                texture_type,
                array_size,
            ) in self.texture_variables:
                declaration = self.format_resource_parameter(
                    texture_type, texture_variable.name, array_size
                )
                resource_params.append(f"{declaration} [[texture({i})]]")
        if self.structured_buffer_variables:
            for (
                buffer_variable,
                i,
                buffer_type,
                array_size,
            ) in self.structured_buffer_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                declaration = self.format_structured_buffer_parameter(
                    buffer_type, buffer_name, array_size
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
                declaration = self.format_glsl_buffer_block_parameter(
                    block, buffer_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.sampler_variables:
            for sampler_variable, i, array_size in self.sampler_variables:
                sampler_name = getattr(sampler_variable, "name", None)
                if dependencies is not None and sampler_name not in dependencies:
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

    def global_resource_parameter_names(self):
        names = set()
        for texture_variable, _, _, _ in self.texture_variables:
            if getattr(texture_variable, "name", None):
                names.add(texture_variable.name)
        for buffer_variable, _, _, _ in self.structured_buffer_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
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
        if isinstance(stmt, VariableNode):
            var_type = self.local_variable_declared_type(stmt)
            self.local_variable_types[stmt.name] = var_type
            if self.is_unsupported_glsl_buffer_block_struct_type(var_type):
                self.current_unsupported_glsl_buffer_block_local_variables.add(
                    stmt.name
                )
                return (
                    f"{indent_str}"
                    f"{self.unsupported_glsl_buffer_block_local_variable_placeholder('Metal', var_type, stmt.name)};\n"
                )

            declaration = format_c_style_array_declaration(
                self.map_type(var_type), stmt.name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            initial_value = getattr(stmt, "initial_value", None)
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    stmt.name,
                    var_type,
                    indent,
                    "Metal",
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
            # Improved array node handling
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)

            if size is None:
                # Dynamic arrays in Metal need a size, use a large enough buffer
                return f"{indent_str}device array<{element_type}, 1024> {stmt.name};\n"
            else:
                return f"{indent_str}array<{element_type}, {size}> {stmt.name};\n"
        elif isinstance(stmt, AssignmentNode):
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
        elif isinstance(stmt, ReturnNode):
            if getattr(stmt, "value", None) is None:
                return f"{indent_str}return;\n"
            if isinstance(stmt.value, list):
                # Multiple return values
                code = ""
                for i, return_stmt in enumerate(stmt.value):
                    code += f"{self.generate_expression(return_stmt)}"
                    if i < len(stmt.value) - 1:
                        code += ", "
                return f"{indent_str}return {code};\n"
            else:
                # Single return value
                return_wrapper = self.current_function_return_wrapper
                if return_wrapper is not None:
                    value = self.generate_expression_with_expected(
                        stmt.value, return_wrapper["source_type"]
                    )
                    return (
                        f"{indent_str}return {return_wrapper['struct_name']}"
                        f"{{{value}}};\n"
                    )
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
            terminator = "" if line.endswith((";", "}")) else ";"
            result += f"{indent_str}{line}{terminator}\n"
        return result

    def local_variable_declared_type(self, stmt):
        var_type = getattr(stmt, "var_type", None)
        if var_type is None:
            var_type = getattr(stmt, "vtype", None)
        if var_type is None:
            var_type = self.expression_result_type(getattr(stmt, "initial_value", None))
        return self.type_name_string(var_type) or "float"

    def local_variable_qualifier(self, node):
        return "const " if "const" in getattr(node, "qualifiers", []) else ""

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
        if isinstance(expr, AssignmentNode):
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
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(
                expr.object
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.object)
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
            if func_name in {
                "float",
                "half",
                "float16",
                "min16float",
                "min10float",
                "double",
                "int",
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
                "f16vec2",
                "f16vec3",
                "f16vec4",
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
        if hasattr(stmt, "expression"):
            if isinstance(stmt.expression, AssignmentNode):
                return self.generate_assignment(stmt.expression)
            expr = self.generate_expression(stmt.expression)
            return expr
        else:
            # Fallback for direct expression
            return self.generate_expression(stmt)

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            target = node.target
            rhs = self.generate_expression_with_expected(
                node.value, self.expression_result_type(node.target)
            )
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            target = node.left
            rhs = self.generate_expression_with_expected(
                node.right, self.expression_result_type(node.left)
            )
            op = getattr(node, "operator", "=")

        block_store = self.generate_glsl_buffer_block_store(target, rhs, op)
        if block_store is not None:
            return block_store
        unsupported_store = self.unsupported_glsl_buffer_block_assignment_diagnostic(
            target
        )
        if unsupported_store is not None:
            return unsupported_store

        lhs = self.generate_expression(target)
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        code += self.generate_scoped_statement_body(if_body, indent + 1)

        code += f"{indent_str}}}"

        # Handle else branch - check if it's another if statement (else-if chain)
        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            # Check if else branch is another IfNode (else-if chain)
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate else if by recursively generating the nested if with else if prefix
                elif_condition = self.generate_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f" else if ({elif_condition}) {{\n"

                # Generate elif body
                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                code += self.generate_scoped_statement_body(elif_body, indent + 1)

                code += f"{indent_str}}}"

                # Recursively handle any remaining else-if chain
                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another else if - recursively handle
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
                        # Final else clause
                        code += " else {\n"
                        code += self.generate_scoped_statement_body(
                            nested_else, indent + 1
                        )
                        code += f"{indent_str}}}"
            else:
                # Regular else clause
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
        if is_switch_lowerable_match(node):
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "Metal")

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

    def generate_expression(self, expr):
        """Render a CrossGL AST expression into Metal expression syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, VariableNode):
            # Fix infinite recursion - directly return the name
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            if hasattr(expr, "name"):
                return enum_value_expression(self, expr.name)
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"{left} {self.map_operator(expr.op)} {right}"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            return f"{self.map_operator(expr.op)}{operand}"
        elif isinstance(expr, WaveOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, RayTracingOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, MeshOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, RayQueryOpNode):
            query = self.generate_expression(expr.query_expr)
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{query}.{expr.operation}({args})"
        elif isinstance(expr, ArrayAccessNode):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_array_load(expr)
            if block_load is not None:
                return block_load
            # Handle array access
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
            return str(expr)
        elif isinstance(expr, FunctionCallNode):
            # Resolve callee expression (can be Identifier/Member/Array access)
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

            static_generic_call = generate_static_generic_numeric_call(self, func_name)
            if static_generic_call is not None:
                return static_generic_call

            unsupported_call = self.unsupported_glsl_buffer_block_function_call(
                func_name
            )
            if unsupported_call is not None:
                return unsupported_call

            enum_constructor = generate_enum_constructor_call(
                self, func_name, expr.args
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
                func_name,
                expr.args,
            )
            if specialized_func_name is not None:
                callee = specialized_func_name
                func_name = specialized_func_name
            # Special handling for common GLSL functions
            if func_name == "normalize":
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"normalize({args})"
            if func_name in ["mix", "clamp", "smoothstep", "step", "dot", "cross"]:
                # These function names are the same in GLSL and Metal
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{func_name}({args})"
            # Vector constructors
            if func_name in [
                "float",
                "half",
                "float16",
                "min16float",
                "min10float",
                "double",
                "int",
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
                "f16vec2",
                "f16vec3",
                "f16vec4",
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
                # Map to Metal's float2, float3, float4
                metal_type = self.map_type(func_name)
                args = ", ".join(
                    self.generate_expression_with_expected(arg, None)
                    for arg in expr.args
                )
                return f"{metal_type}({args})"
            # Standard function call
            self.validate_function_image_access_arguments(func_name, expr.args)
            args = [self.generate_expression(arg) for arg in expr.args]
            if func_name in self.user_function_names:
                args.extend(
                    self.cbuffer_parameter_name(cbuffer)
                    for cbuffer in self.required_function_cbuffers(func_name)
                )
                args.extend(self.required_function_resource_argument_names(func_name))
            args = ", ".join(args)
            return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_member_load(expr)
            if block_load is not None:
                return block_load
            obj = self.generate_expression_with_expected(expr.object, None)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition)} ? {self.generate_expression(expr.true_expr)} : {self.generate_expression(expr.false_expr)}"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            # Handle LiteralNode
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
            # Handle IdentifierNode
            name = getattr(expr, "name", str(expr))
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
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
            return (
                "/* unsupported Metal buffer append: requires explicit "
                "counter buffer */"
            )
        if func_name == "buffer_consume" and args:
            return (
                "0 /* unsupported Metal buffer consume: requires explicit "
                "counter buffer */"
            )
        if func_name == "buffer_dimensions" and args:
            diagnostic = (
                "0 /* unsupported Metal buffer dimensions: device buffers do not "
                "carry length */"
            )
            if len(args) >= 2:
                target = self.generate_expression(args[1])
                return f"{target} = {diagnostic}"
            return diagnostic
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

    def format_structured_buffer_parameter(self, vtype, name, array_size=None):
        element_type = self.structured_buffer_element_type(vtype)
        address_space = self.structured_buffer_address_space(vtype)
        pointer_type = f"{address_space} {element_type}*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type} {name}"

    def format_glsl_buffer_block_parameter(self, block, name, array_size=None):
        address_space = "const device" if block.get("readonly") else "device"
        pointer_type = f"{address_space} uchar*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type} {name}"

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) == "sampler"

    def is_resource_parameter_type(self, vtype):
        return self.is_structured_buffer_type(vtype) or self.resource_base_type(
            vtype
        ) in {
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

    def is_texture_or_image_resource_type(self, vtype):
        return (
            self.is_resource_parameter_type(vtype)
            and not self.is_sampler_type(vtype)
            and not self.is_structured_buffer_type(vtype)
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
            if texture_type.startswith("texture1d_array<"):
                query_lod_coordinate_dimension = 2
            elif texture_type.startswith("texture1d<"):
                query_lod_coordinate_dimension = 1
            elif texture_type.startswith(("texture2d_array<", "depth2d_array<")):
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
        if semantic:
            return self.map_semantic(semantic)
        if shader_type in {"vertex", "fragment", "compute"}:
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
        elif self.is_structured_buffer_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_texture_or_image_resource_type(raw_param_type):
            namespace = "texture"
            attribute_names = {"binding", "texture"}
            prefixes = ("t", "u")
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
        for buffer_variable, binding, _, _ in self.structured_buffer_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                getattr(buffer_variable, "name", "<anonymous>"),
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
        return ""

    def format_parameter_declaration(
        self, raw_param_type, mapped_type, name, node=None
    ):
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
                raw_param_type, name, array_size
            )
        array_type = self.resource_array_parameter(raw_param_type, node)
        if array_type is not None:
            resource_type, array_size = array_type
            return self.format_resource_parameter(resource_type, name, array_size)
        return format_c_style_array_declaration(mapped_type, name)

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
        return collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_resource_globals(ast),
            function_arrays=self.collect_unsized_resource_parameters(ast),
            fixed_global_array_sizes=self.collect_fixed_resource_global_sizes(ast),
            fixed_function_array_sizes=self.collect_fixed_resource_parameter_sizes(ast),
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
        return {"geometry", "tessellation_control", "tessellation_evaluation"}

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

    def stage_entry_types(self):
        return {
            "vertex",
            "fragment",
            "compute",
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
        if stage_name == "vertex":
            return f"vertex_{func_name}"
        if stage_name == "fragment":
            return f"fragment_{func_name}"
        if stage_name in {"compute", "ray_generation"}:
            return f"kernel_{func_name}"
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

    def stage_entry_names(self, ast, target_stage=None):
        stage_entry_types = self.stage_entry_types()
        entries = collect_stage_entry_records(ast, target_stage, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, target_stage, stage_entry_types
        )
        return assign_stage_entry_names(entries, used_names, self.stage_entry_base_name)

    def all_functions(self, ast):
        functions = list(getattr(ast, "functions", []) or [])
        for stage in getattr(ast, "stages", {}).values():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
            functions.extend(getattr(stage, "local_functions", []) or [])
        return functions

    def collect_function_parameters(self, functions):
        parameters = []
        for func in functions or []:
            parameters.extend(getattr(func, "parameters", getattr(func, "params", [])))
        return parameters

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
        return self.is_resource_parameter_type(
            vtype
        ) or self.is_glsl_buffer_block_variable(node, vtype)

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
        buffer_names = (
            self.global_structured_buffer_names()
            | self.global_glsl_buffer_block_names()
        )
        sampler_names = self.global_sampler_names()
        dependencies = set()

        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if hasattr(node, "__class__") and "Identifier" in str(node.__class__):
                name = getattr(node, "name", None)
                if (
                    name
                    and name not in local_names
                    and (
                        name in texture_names
                        or name in buffer_names
                        or name in sampler_names
                    )
                ):
                    dependencies.add(name)

            if isinstance(node, FunctionCallNode):
                self.add_texture_call_resource_dependencies(
                    node, local_names, texture_names, sampler_names, dependencies
                )
                self.add_buffer_call_resource_dependencies(
                    node, local_names, buffer_names, dependencies
                )

        return dependencies

    def add_texture_call_resource_dependencies(
        self, call, local_names, texture_names, sampler_names, dependencies
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

    def global_structured_buffer_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _ in self.structured_buffer_variables
            if getattr(buffer_variable, "name", None)
        }

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

    def required_function_resource_argument_names(self, func_name):
        return (
            [
                texture_variable.name
                for texture_variable, _, _ in self.required_function_textures(func_name)
            ]
            + [
                buffer_variable.name
                for buffer_variable, _, _ in self.required_function_structured_buffers(
                    func_name
                )
            ]
            + [
                buffer_variable.name
                for buffer_variable, _, _ in self.required_function_glsl_buffer_blocks(
                    func_name
                )
            ]
            + [
                sampler_variable.name
                for sampler_variable, _ in self.required_function_samplers(func_name)
            ]
        )

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
        if hasattr(value, "name"):
            return str(value.name)
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
                or self.is_glsl_buffer_block_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

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
                or self.is_glsl_buffer_block_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

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
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
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
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
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
        if not var_name:
            return None
        block = self.current_glsl_buffer_block_parameters.get(
            var_name
        ) or self.lowered_glsl_buffer_blocks.get(var_name)
        if block is None:
            return None
        buffer_expr = self.generate_expression(object_expr)
        member_name = getattr(expr, "member", None)
        member = block["members"].get(member_name)
        if member is None:
            return None
        return {
            "buffer": buffer_expr,
            "member": member_name,
            "readonly": block["readonly"],
            **member,
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
        return {**member, "offset_expr": offset}

    def metal_scalar_load(self, component_type, buffer_name, offset):
        return (
            f"(*reinterpret_cast<const device {component_type}*>"
            f"({buffer_name} + {offset}))"
        )

    def metal_scalar_store(self, component_type, buffer_name, offset, value):
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
        return self.metal_buffer_load(access["buffer"], access["offset"], access)

    def generate_glsl_buffer_block_array_load(self, expr):
        access = self.glsl_buffer_block_array_access(expr)
        if access is None:
            return None
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
            if hasattr(node.var_type, "name") or hasattr(node.var_type, "element_type"):
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
        elif self.is_structured_buffer_type(vtype):
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

    def reserve_explicit_global_resource_bindings(self, global_vars, used_bindings):
        for node in global_vars:
            metadata = self.global_resource_binding_metadata(node)
            if metadata is None:
                continue
            namespace, binding, resource_count, var_name = metadata
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                namespace,
                binding,
                resource_count,
                var_name,
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

    def reserve_cbuffer_bindings(self, used_bindings):
        buffer_index = 0
        for cbuffer in self.cbuffer_variables:
            binding = self.explicit_resource_binding_index(
                cbuffer, {"binding", "buffer"}, ("b",)
            )
            if binding is None:
                continue
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                1,
                getattr(cbuffer, "name", "<anonymous>"),
            )
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
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                1,
                getattr(cbuffer, "name", "<anonymous>"),
            )
            self.cbuffer_binding_indices[id(cbuffer)] = binding
            buffer_index = max(buffer_index, binding + 1)
        return buffer_index

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

    def texture_sampler_expression(self, texture_name):
        sampler_arg = ""
        for sampler_variable, _, _ in self.sampler_variables:
            if sampler_variable.name == texture_name + "Sampler":
                sampler_arg = sampler_variable.name
                break
        return sampler_arg or self.default_sampler_expression()

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
            else self.texture_sampler_expression(texture_base_name)
        )
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, sampler_arg, coord, extra_args

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        return self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        arg_type = self.expression_result_type(texture_arg)
        if arg_type is None or not self.is_texture_or_image_resource_type(arg_type):
            return None
        return self.map_resource_type_with_format(self.resource_base_type(arg_type))

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
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )

    def is_array_texture_resource(self, texture_type):
        return texture_type in {
            "texture1d_array<float>",
            "texture2d_array<float>",
            "depth2d_array<float>",
            "texturecube_array<float>",
            "depthcube_array<float>",
        }

    def is_multisample_texture_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {
            "texture2d_ms<float>",
            "texture2d_ms_array<float>",
        }

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
        if texture_type == "texture1d_array<float>":
            coord_x = self.vector_component(coord, "x")
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer
        if texture_type in {"texturecube_array<float>", "depthcube_array<float>"}:
            return self.cube_array_texture_coordinate_parts(coord)
        return self.array_texture_coordinate_parts(coord)

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("Metal", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

    def texture_gradient_options(self, texture_type, ddx, ddy):
        if texture_type in {
            "texturecube<float>",
            "depthcube<float>",
            "texturecube_array<float>",
            "depthcube_array<float>",
        }:
            return f"gradientcube({ddx}, {ddy})"
        if texture_type == "texture3d<float>":
            return f"gradient3d({ddx}, {ddy})"
        return f"gradient2d({ddx}, {ddy})"

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        color_offset_types = {"texture2d<float>", "texture2d_array<float>"}
        depth_offset_types = {"depth2d<float>", "depth2d_array<float>"}
        return {
            "texture_type": texture_type,
            "gather": (
                texture_type
                in {
                    "texture2d<float>",
                    "texture2d_array<float>",
                    "texturecube<float>",
                    "texturecube_array<float>",
                }
            ),
            "gather_offset": texture_type in color_offset_types,
            "sample_offset": texture_type in color_offset_types,
            "projected_offset": texture_type in color_offset_types,
            "compare_offset": texture_type in depth_offset_types,
            "gather_compare_offset": texture_type in depth_offset_types,
        }

    def texture_gather_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_sample_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

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
        texture_type = self.texture_resource_type(texture_arg)
        return {
            "texture_type": texture_type,
            "storage_image": self.is_storage_image_resource(texture_type),
            "multisample": (
                self.is_multisample_texture_resource(texture_type)
                or self.is_multisample_storage_image_resource(texture_type)
            ),
            "size_descriptor": self.texture_query_size_descriptor(texture_type),
        }

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        query_descriptor = self.texture_query_resource_descriptor(texture_arg)
        size_descriptor = query_descriptor["size_descriptor"]
        if size_descriptor is None:
            return None

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
        return storage_image_load_component_suffix(
            image_format,
            expected_scalar=self.is_scalar_value_type(
                self.current_expression_expected_type
            ),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                image_type, image_format
            ),
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
        return storage_image_store_value_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                image_type, image_format
            ),
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

    def validate_image_store_value_shape(self, image_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(image_type, image_format)
        value_channels = image_store_value_shape_mismatch(
            expected_channels, self.expression_component_count(value_arg)
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
        texture_type = self.texture_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        is_array_texture = self.is_array_texture_resource(texture_type)
        if is_array_texture:
            coord_xy, layer = self.texture_coordinate_parts(texture_type, coord)

        if is_texture_sample_operation(
            func_name
        ) and self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)

        if is_texture_sample_basic_operation(func_name):
            if extra_args:
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
            lod = self.generate_expression(extra_args[0])
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer}, level({lod}))"
            return f"{texture_name}.sample({sampler_arg}, {coord}, level({lod}))"
        if is_texture_sample_grad_operation(func_name) and len(extra_args) >= 2:
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
            lod_coord = self.texture_query_lod_coordinate(texture_type, coord)
            return (
                f"float2({texture_name}.calculate_unclamped_lod({sampler_arg}, {lod_coord}), "
                f"{texture_name}.calculate_clamped_lod({sampler_arg}, {lod_coord}))"
            )
        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            lod = self.generate_expression(args[2])
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource(texture_type):
                if texture_type == "texture2d_ms_array<float>":
                    texel_xy, layer = self.array_texture_coordinate_parts(coord)
                    return f"{texture_name}.read({texel_xy}, {layer}, uint({lod}))"
                return f"{texture_name}.read({coord}, uint({lod}))"
            if is_array_texture:
                texel_coord, layer = self.texture_coordinate_parts(texture_type, coord)
                return f"{texture_name}.read({texel_coord}, {layer}, {lod})"
            return f"{texture_name}.read({coord}, {lod})"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            lod = self.generate_expression(args[2])
            offset = self.generate_expression(args[3])
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource(texture_type):
                return unsupported_multisample_texel_fetch_offset_expression("Metal")
            if is_array_texture:
                texel_coord, layer = self.texture_coordinate_parts(texture_type, coord)
                return (
                    f"{texture_name}.read(({texel_coord} + {offset}), {layer}, {lod})"
                )
            return f"{texture_name}.read(({coord} + {offset}), {lod})"

        return None

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

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

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
            mapped_semantic = self.semantic_map.get(semantic, semantic)
            # If the mapped semantic already has brackets, use it as-is
            if mapped_semantic.startswith("[[") and mapped_semantic.endswith("]]"):
                return f" {mapped_semantic}"
            else:
                # Add brackets for Metal attribute syntax
                return f" [[{mapped_semantic}]]"
        else:
            return ""
